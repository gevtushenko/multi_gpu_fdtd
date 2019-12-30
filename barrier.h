//
// Created by egi on 12/30/19.
//

#ifndef MULTIGPUFDTD_BARRIER_H
#define MULTIGPUFDTD_BARRIER_H

#include <atomic>
#include <xmmintrin.h>

class barrier_class
{
public:
  barrier_class () = delete;
  explicit barrier_class (int threads_count_arg)
    : threads_count (threads_count_arg)
    , barrier_epoch (0)
    , threads_in_barrier (0)
  { }

  void operator ()()
  {
    if (threads_count == 1)
      return;

    const unsigned int thread_epoch = barrier_epoch.load ();

    if (threads_in_barrier.fetch_add (1) == threads_count - 1)
      {
        threads_in_barrier.store (0);
        barrier_epoch.fetch_add (1);
      }
    else
      {
        while (thread_epoch == barrier_epoch.load ())
          {
            _mm_pause ();
          }
      }
  }

private:
  int threads_count {};
  std::atomic<unsigned int> barrier_epoch;
  std::atomic<unsigned int> threads_in_barrier;
};

class thread_info_class
{
public:
  thread_info_class () = delete;
  thread_info_class (int thread_id_arg, int threads_count_arg, barrier_class &barrier_arg)
    : thread_id (thread_id_arg)
    , threads_count (threads_count_arg)
    , barrier (barrier_arg)
  {

  }

  void sync () const
  {
    barrier ();
  }

public:
  const int thread_id {};
  const int threads_count {};

private:
  barrier_class &barrier;
};


#endif //MULTIGPUFDTD_BARRIER_H
