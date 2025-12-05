#include "ThreadSafeQueue.hpp"

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
std::optional<T> ThreadSafeQueue<T>::pop() {
    // unique_lock is required for condition_variable::wait
    std::unique_lock<std::mutex> lock(mtx);

    // Wait until queue is not empty OR shutdown is triggered.
    // The lambda handles the spurious wakeups.
    cv.wait(lock, [this] { 
        return !queue.empty() || shutdown_flag; 
    });

    // If the queue is empty and shutdown is flagged, return nullopt (terminate signal)
    if (queue.empty() && shutdown_flag) {
        return std::nullopt;
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