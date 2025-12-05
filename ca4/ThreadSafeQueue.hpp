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
    std::optional<T> pop();

    // Checks if the queue is empty
    bool empty() const;

    // Stops the queue and notifies all waiting threads
    void shutdown();
};

#include "ThreadSafeQueue.cpp"

#endif // THREAD_SAFE_QUEUE_HPP