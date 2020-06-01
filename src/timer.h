#include <chrono>

#define START_TIMER std::chrono::high_resolution_clock::now();
#define END_TIMER   std::chrono::high_resolution_clock::now();

typedef std::chrono::time_point<std::chrono::high_resolution_clock> TimePoint;  
typedef std::chrono::duration<double> Time;

Time elapsedTime;
TimePoint start, end;

void startTimer() {
  start = START_TIMER
}

void stopTimer() {
  end = END_TIMER
  elapsedTime += (end - start);
}

double getTimer() {
  return elapsedTime.count();
}
