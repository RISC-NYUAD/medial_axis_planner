#ifndef __TIMER_HEADER__
#define __TIMER_HEADER__

#include <chrono>

class Timer {
  // A utility timer class
 public:
  void Tic() { current_time_ = std::chrono::high_resolution_clock::now(); };
  float Toc() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now - current_time_);
    this->Tic();
    return static_cast<float>(duration.count()) / 1e9;
  }

 private:
  std::chrono::high_resolution_clock::time_point current_time_;
};

#endif
