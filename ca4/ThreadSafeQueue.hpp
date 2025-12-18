#ifndef THREAD_SAFE_QUEUE_HPP
#define THREAD_SAFE_QUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <optional> // C++17

template <typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue;
    mutable std::mutex mtx;             // Mutex for mutual exclusion
    std::condition_variable cv;         // Condition variable for waiting/notifying
    std::atomic<bool> shutdown_flag;    // Atomic flag for shutdown signaling

public:
    // Constructor
    ThreadSafeQueue();

    // Destructor (calls shutdown)
    ~ThreadSafeQueue();

    // Delete copy constructor and assignment operator for safety
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    // Pushes a value into the queue
    void push(T value);

    // Pops a value from the queue. Returns std::nullopt if shutdown.
    T pop();

    // Checks if the queue is empty
    bool empty() const;

    // Stops the queue and notifies all waiting threads
    void shutdown();
};

template <typename T>
ThreadSafeQueue<T>::ThreadSafeQueue() : shutdown_flag(false) {}

template <typename T>
ThreadSafeQueue<T>::~ThreadSafeQueue() {
    shutdown(); // Ensure all waiting threads are released upon destruction
}

template <typename T>
void ThreadSafeQueue<T>::push(T value) {
    // lock_guard is used for simple critical sections (RAII)
    std::lock_guard<std::mutex> lock(mtx);
    
    // If shutdown has started, we generally shouldn't accept new items,
    // though this depends on specific requirements. Here we ignore push if shutdown.
    if (shutdown_flag) return;

    queue.push(std::move(value));
    
    // Notify one waiting thread that data is ready
    cv.notify_one(); 
}

template <typename T>
T ThreadSafeQueue<T>::pop() {
    // unique_lock is required for condition_variable::wait
    std::unique_lock<std::mutex> lock(mtx);

    // Wait until queue is not empty OR shutdown is triggered.
    // The lambda handles the spurious wakeups.
    cv.wait(lock, [this] { 
        return !queue.empty() || shutdown_flag; 
    });

    // If the queue is empty and shutdown is flagged, return nullopt (terminate signal)
    if (queue.empty() && shutdown_flag) {
        return T();
    }

    // Process the item
    T value = std::move(queue.front());
    queue.pop();
    
    return value;
}

template <typename T>
bool ThreadSafeQueue<T>::empty() const {
    std::lock_guard<std::mutex> lock(mtx);
    return queue.empty();
}

template <typename T>
void ThreadSafeQueue<T>::shutdown() {
    {
        // Scope block to minimize lock duration
        std::lock_guard<std::mutex> lock(mtx);
        shutdown_flag = true;
    }
    
    // Notify ALL threads to wake up and check the shutdown_flag
    cv.notify_all();
}

#endif // THREAD_SAFE_QUEUE_HPP