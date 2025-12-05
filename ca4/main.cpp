#include <iostream>
#include <vector>
#include <string>
#include <pthread.h> // Include POSIX Threads
#include <unistd.h>  // For sleep()
#include "ThreadSafeQueue.hpp"
#include <opencv2/opencv.hpp>

// Define data types
using RawData = std::string;
using ResultData = std::string;

// --- Argument Structures ---
// Since pthread_create only accepts a single void* argument,
// we need structs to bundle the queues required for each worker.

struct InputArgs {
    ThreadSafeQueue<cv::Mat>* raw_queue;
    int camera_id; // Added camera ID to make it flexible
};

struct ProcessArgs {
    ThreadSafeQueue<RawData>* raw_queue;
    ThreadSafeQueue<ResultData>* result_queue;
};

struct OutputArgs {
    ThreadSafeQueue<ResultData>* result_queue;
};

// --- Worker Functions ---

// 1. Input Worker
void* input_worker(void* args) {
    // 1. Cast the void* argument back to InputArgs*
    InputArgs* my_args = static_cast<InputArgs*>(args);
    
    // 2. Open the webcam (0 is usually the default camera)
    cv::VideoCapture cap(my_args->camera_id);

    // Check if camera opened successfully
    if (!cap.isOpened()) {
        std::cerr << "[Input] Error: Could not open camera." << std::endl;
        return NULL;
    }

    std::cout << "[Input] Camera opened. Starting capture..." << std::endl;

    cv::Mat frame;
    while (true) {
        // 3. Read a new frame from video capture
        cap >> frame;

        // 4. Check if we succeeded (if frame is empty, stream might have ended)
        if (frame.empty()) {
            my_args->raw_queue->push();
            std::cerr << "[Input] Error: Blank frame grabbed or camera disconnected." << std::endl;
            break;
        }

        // 5. Push the frame to the thread-safe queue
        // Note: frame.clone() is often safer in threading to ensure 
        // a deep copy of image data is passed, preventing data races 
        // if OpenCV reuses the memory of 'frame' in the next iteration.
        my_args->raw_queue->push(frame.clone());

        // Optional: Log every X frames to avoid console spam
        // std::cout << "[Input] Pushed frame to queue." << std::endl;
    }

    // 6. Cleanup
    // When the loop breaks, we signal that input is done.
    // However, the actual queue shutdown usually happens in main() after join.
    std::cout << "[Input] Capturing stopped." << std::endl;
    
    // Release the camera explicitly (optional, destructor does it too)
    cap.release();

    return NULL;
}

// 2. Process Worker
void* process_worker(void* args) {
    ProcessArgs* my_args = static_cast<ProcessArgs*>(args);

    std::cout << "[Process] Worker started (Sobel Algo)." << std::endl;

    while (true) {
        // 1. Receive frame from queue
        cv::Mat frame = my_args->raw_queue->pop();

        // Check for stop signal (Poison Pill)
        if (frame.empty()) {
            std::cout << "[Process] Stop signal received." << std::endl;
            // Forward the empty frame to the next queue to stop the output worker
            my_args->result_queue->push(frame);
            break;
        }

        // 2. Convert to Grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // 3. Apply Sobel Edge Detection
        cv::Mat grad_x, grad_y;
        cv::Mat abs_grad_x, abs_grad_y;
        cv::Mat detected_edges;

        // Gradient X
        // ddepth = CV_16S to avoid overflow/negative values issues
        cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
        
        // Gradient Y
        cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

        // Converting back to CV_8U (Absolute value)
        cv::convertScaleAbs(grad_x, abs_grad_x);
        cv::convertScaleAbs(grad_y, abs_grad_y);

        // 4. Combine X and Y edges
        // Total Gradient (approximate) = 0.5 * |grad_x| + 0.5 * |grad_y|
        cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, detected_edges);

        // 5. Send to result queue
        my_args->result_queue->push(detected_edges);
    }

    std::cout << "[Process] Worker stopped." << std::endl;
    return NULL;
}

// 3. Output Worker
void* output_worker(void* args) {
    OutputArgs* my_args = static_cast<OutputArgs*>(args);

    while (true) {
        auto result_opt = my_args->result_queue->pop();
        
        if (!result_opt.has_value()) {
            std::cout << "[Output] Shutdown signal received." << std::endl;
            break;
        }

        std::string result = result_opt.value();
        std::cout << "[Output] Final Result: " << result << std::endl;
    }
    return NULL;
}

// --- Main Application ---

int main() {
    // Queues
    ThreadSafeQueue<RawData> raw_queue;
    ThreadSafeQueue<ResultData> result_queue;

    std::cout << "Starting Pipeline with Pthreads..." << std::endl;

    // Thread identifiers
    pthread_t producer_thread;
    pthread_t processor_thread;
    pthread_t consumer_thread;

    // Prepare arguments
    InputArgs in_args = { &raw_queue };
    ProcessArgs proc_args = { &raw_queue, &result_queue };
    OutputArgs out_args = { &result_queue };

    // Create Threads
    // Syntax: pthread_create(&thread_id, attributes, function_ptr, argument_ptr)
    if (pthread_create(&producer_thread, NULL, input_worker, (void*)&in_args) != 0) {
        std::cerr << "Failed to create input thread" << std::endl;
        return 1;
    }

    if (pthread_create(&processor_thread, NULL, process_worker, (void*)&proc_args) != 0) {
        std::cerr << "Failed to create process thread" << std::endl;
        return 1;
    }

    if (pthread_create(&consumer_thread, NULL, output_worker, (void*)&out_args) != 0) {
        std::cerr << "Failed to create output thread" << std::endl;
        return 1;
    }

    // Wait for Input thread to finish
    pthread_join(producer_thread, NULL);

    // Shutdown raw queue to signal processor to stop
    std::cout << "Input finished. Shutting down Raw Queue..." << std::endl;
    raw_queue.shutdown();

    // Wait for Processor thread to finish
    pthread_join(processor_thread, NULL);

    // Shutdown result queue to signal output to stop
    std::cout << "Processing finished. Shutting down Result Queue..." << std::endl;
    result_queue.shutdown();

    // Wait for Consumer thread to finish
    pthread_join(consumer_thread, NULL);

    std::cout << "Pipeline finished successfully." << std::endl;

    return 0;
}