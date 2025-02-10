#pragma once

#include <chrono>
#include <iostream>
#include <string>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    void stop(const std::string& timer_name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << timer_name << ": " << duration/1000.0 << "ms" << std::endl;
    }
}; 