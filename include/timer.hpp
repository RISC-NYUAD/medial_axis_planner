#pragma once

#include <chrono>

class Timer {
  // A utility timer class
public:
  void tic() { current_time_ = std::chrono::high_resolution_clock::now(); };
  double toc() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now - current_time_);
    return static_cast<double>(duration.count()) / 1e9;
  }

private:
  std::chrono::high_resolution_clock::time_point current_time_;
};
