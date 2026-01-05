#include <iostream>
#include <vector>
#include <string>
#include <pthread.h> // Include POSIX Threads
#include <unistd.h>  // For sleep()
#include "ThreadSafeQueue.hpp"
#include <opencv2/opencv.hpp>
// #include <conio.h>

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>


using namespace std;
using namespace cv;
// --- Argument Structures ---
// Since pthread_create only accepts a single void* argument,
// we need structs to bundle the queues required for each worker.

struct InputArgs {
    ThreadSafeQueue<cv::Mat>* raw_queue;
    int camera_id; // Added camera ID to make it flexible
    double* shared_fps;
};

struct ProcessArgs {
    ThreadSafeQueue<cv::Mat>* raw_queue;
    ThreadSafeQueue<cv::Mat>* result_queue;
};

struct OutputArgs {
    ThreadSafeQueue<cv::Mat>* result_queue;
    std::string filename;
    double* shared_fps;
};

int kbhit_linux() {
    termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}

// --- Worker Functions ---

// 1. Input Worker
void* input_worker(void* args) {
    // 1. Cast the void* argument back to InputArgs*
    InputArgs* my_args = static_cast<InputArgs*>(args);
    
    // 2. Open the webcam (0 is usually the default camera)
    cv::VideoCapture cap(my_args->camera_id);
    // cv::VideoCapture cap("sample/test.avi");

    // Check if camera opened successfully
    if (!cap.isOpened()) {
        std::cerr << "[Input] Error: Could not open camera." << std::endl;
        return NULL;
    }
    std::cout << "[Input] Camera opened. Starting capture..." << std::endl;

    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    if (actual_fps <= 0) actual_fps = 30.0;
    *(my_args->shared_fps) = actual_fps;
    
    std::cout << "[Input] Camera FPS detected: " << actual_fps << std::endl;

    cv::Mat frame;
    while (true) {
        // 3. Read a new frame from video capture
        cap >> frame;

        // 4. Check if we succeeded (if frame is empty, stream might have ended)
        if (frame.empty()) {
            std::cerr << "[Input] Error: Blank frame grabbed or camera disconnected." << std::endl;
            // send a signal to worker using empty frame
            my_args->raw_queue->push(cv::Mat());
            break;
        }

        // 5. Push the frame to the thread-safe queue
        // Note: frame.clone() is often safer in threading to ensure 
        // a deep copy of image data is passed, preventing data races 
        // if OpenCV reuses the memory of 'frame' in the next iteration.
        my_args->raw_queue->push(frame.clone());

        // Optional: Log every X frames to avoid console spam
        // std::cout << "[Input] Pushed frame to queue." << std::endl;

        if (kbhit_linux()) {
            int key = getchar();
            if (key == 'q' || key == 27) {
                cout << "Stopping recording..." << endl;
                my_args->raw_queue->push(cv::Mat());
                break;
            }
        }
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
    
    cv::VideoWriter writer;
    bool is_writer_initialized = false;
    long frame_count = 0;
    
    // Timer variables
    auto start_time = std::chrono::steady_clock::now();

    std::cout << "[Output] Worker started. Waiting for frames..." << std::endl;

    while (true) {
        // 1. Receive frame from queue
        cv::Mat frame = my_args->result_queue->pop();

        // Check for stop signal
        if (frame.empty()) {
            std::cout << "[Output] Stop signal received. Finalizing..." << std::endl;
            break;
        }

        // 2. Initialize VideoWriter with the first frame
        if (!is_writer_initialized) {
            // Using MJPG codec
            int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

            double fps = *(my_args->shared_fps); 
            if (fps <= 0) fps = 30.0;

            cv::Size frame_size = frame.size();
            
            // IMPORTANT: isColor = false because Sobel output is Grayscale
            bool isColor = false; 

            writer.open(my_args->filename, fourcc, fps, frame_size, isColor);

            if (!writer.isOpened()) {
                std::cerr << "[Output] Error: Could not open video writer!" << std::endl;
                break; // Exit if we can't save
            }

            // Start the timer when the first frame arrives
            start_time = std::chrono::steady_clock::now();
            is_writer_initialized = true;
            std::cout << "[Output] VideoWriter initialized. Recording..." << std::endl;
        }

        // 3. Save processed frame
        writer.write(frame);
        frame_count++;
        
        // Optional: Show progress every 30 frames
        if (frame_count % 30 == 0) {
            std::cout << "." << std::flush;
        }
    }

    // 4. Calculate Average FPS
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    if (elapsed_seconds.count() > 0) {
        double avg_fps = frame_count / elapsed_seconds.count();
        std::cout << "\n[Output] Processing Finished." << std::endl;
        std::cout << "[Output] Total Frames: " << frame_count << std::endl;
        std::cout << "[Output] Time:  " << elapsed_seconds.count() << std::endl;
        std::cout << "[Output] Average FPS:  " << avg_fps << std::endl;
    }

    // Release writer
    writer.release();
    return NULL;
}

// --- Main Application ---

int main() {
    // 1. Initialize thread-safe queues
    ThreadSafeQueue<cv::Mat> raw_queue;
    ThreadSafeQueue<cv::Mat> result_queue;

    // 2. Prepare arguments for workers
    double shared_fps = 30.0;
    
    InputArgs input_args;
    input_args.raw_queue = &raw_queue;
    input_args.camera_id = 0; // Default camera index
    input_args.shared_fps = &shared_fps;

    ProcessArgs process_args;
    process_args.raw_queue = &raw_queue;
    process_args.result_queue = &result_queue;

    OutputArgs output_args;
    output_args.result_queue = &result_queue;
    output_args.filename = "output_sobel_serial.avi";
    output_args.shared_fps = &shared_fps;


    // 3. Create threads
    pthread_t thread_in, thread_proc, thread_out;

    std::cout << "[Main] Starting threads..." << std::endl;

    if (pthread_create(&thread_in, NULL, input_worker, &input_args) != 0) {
        std::cerr << "[Main] Error: Failed to create Input thread." << std::endl;
        return 1;
    }

    if (pthread_create(&thread_proc, NULL, process_worker, &process_args) != 0) {
        std::cerr << "[Main] Error: Failed to create Process thread." << std::endl;
        return 1;
    }

    if (pthread_create(&thread_out, NULL, output_worker, &output_args) != 0) {
        std::cerr << "[Main] Error: Failed to create Output thread." << std::endl;
        return 1;
    }

    // 4. Synchronization Logic (Waterfall Shutdown)
    
    // Step A: Wait for Input worker to finish (Camera disconnects or error)
    // Note: input_worker sends an empty frame to raw_queue before exiting.
    pthread_join(thread_in, NULL);
    std::cout << "[Main] Input thread joined." << std::endl;

    // Step B: Wait for Process worker to finish
    // It finishes after receiving the empty frame from input_worker.
    pthread_join(thread_proc, NULL);
    std::cout << "[Main] Process thread joined." << std::endl;

    // Step C: Send stop signal to Output worker
    // As per requirement: Manually push empty frame after process is done.
    std::cout << "[Main] Sending stop signal to Output worker..." << std::endl;
    result_queue.push(cv::Mat());

    // Step D: Wait for Output worker to finish saving and calculating FPS
    pthread_join(thread_out, NULL);
    std::cout << "[Main] Output thread joined." << std::endl;

    std::cout << "[Main] Application finished successfully." << std::endl;


    return 0;
}