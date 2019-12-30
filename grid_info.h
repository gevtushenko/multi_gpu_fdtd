//
// Created by egi on 12/30/19.
//

#ifndef MULTIGPUFDTD_GRID_INFO_H
#define MULTIGPUFDTD_GRID_INFO_H

#include <cuda_runtime.h>
#include <memory>

#include "gpu_utils.h"

class grid_barrier_accessor_class
{
public:
  grid_barrier_accessor_class () = delete;
  grid_barrier_accessor_class (
    cudaEvent_t *push_top_arg,
    cudaEvent_t *push_bottom_arg,
    int device_id_arg,
    int devices_count_arg)
    : device_id (device_id_arg)
    , devices_count (devices_count_arg)
    , push_top (push_top_arg)
    , push_bottom (push_bottom_arg)
  {
    for (int iteration: {0, 1})
      {
        throw_on_error (cudaEventCreateWithFlags (get_top_done (iteration, device_id), cudaEventDisableTiming), __FILE__, __LINE__);
        throw_on_error (cudaEventCreateWithFlags (get_bottom_done (iteration, device_id), cudaEventDisableTiming), __FILE__, __LINE__);
      }
  }

  ~grid_barrier_accessor_class ()
  {
    try {
      for (int iteration: {0, 1})
        {
          throw_on_error (cudaEventDestroy (*get_top_done (iteration, device_id)), __FILE__, __LINE__);
          throw_on_error (cudaEventDestroy (*get_bottom_done (iteration, device_id)), __FILE__, __LINE__);
        }
    } catch (...) {
      std::cerr << "Error in barrier accessor destructor!" << std::endl;
    }
  }

  cudaEvent_t *get_top_done (int iteration, int device_id)
  {
    return get_event (push_top, iteration, device_id);
  }

  cudaEvent_t *get_bottom_done (int iteration, int device_id)
  {
    return get_event (push_bottom, iteration, device_id);
  }

private:
  cudaEvent_t *get_event (cudaEvent_t *events, int iteration, int device_id)
  {
    return events + devices_count * (iteration % 2) + device_id;
  }

private:
  int device_id {};
  int devices_count {};
  cudaEvent_t *push_top;
  cudaEvent_t *push_bottom;
};

class grid_barrier_class
{
public:
  explicit grid_barrier_class (int devices_count_arg)
    : devices_count (devices_count_arg)
    , push_top_done (new cudaEvent_t[2 * devices_count])
    , push_bottom_done (new cudaEvent_t[2 * devices_count])
  {
  }

  grid_barrier_accessor_class create_accessor (int device_id)
  {
    return {
      push_top_done.get (),
      push_bottom_done.get (),
      device_id,
      devices_count
    };
  }

private:
  int devices_count {};
  std::unique_ptr<cudaEvent_t[]> push_top_done;
  std::unique_ptr<cudaEvent_t[]> push_bottom_done;
};

class grid_info_class
{
public:
  grid_info_class () = delete;
  grid_info_class (int process_nx, int process_ny, const thread_info_class &thread_info, grid_barrier_accessor_class &barrier_arg)
    : own_nx (process_nx)
    , barrier (barrier_arg)
  {
    const int chunk_size = process_ny / thread_info.threads_count;
    own_ny = thread_info.thread_id == thread_info.threads_count - 1
       ? chunk_size + process_ny % chunk_size
       : chunk_size;
    row_begin_in_process = chunk_size * thread_info.thread_id;
    row_end_in_process = thread_info.thread_id < thread_info.threads_count - 1
                       ? chunk_size * (thread_info.thread_id + 1)
                       : process_ny;
  }

private:
  int own_nx {};
  int own_ny {};

  int row_begin_in_process {};
  int row_end_in_process {};

  grid_barrier_accessor_class &barrier;
};

#endif //MULTIGPUFDTD_GRID_INFO_H
