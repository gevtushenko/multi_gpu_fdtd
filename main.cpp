#include <thread>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

#include "fdtd.h"
#include "barrier.h"
#include "grid_info.h"
#include "gpu_utils.h"

void prepare_nvlink (const thread_info_class &thread_info)
{
  throw_on_error (cudaSetDevice (thread_info.thread_id), __FILE__, __LINE__);

  bool nvlink_enabled = true;

  for (int other_device_id = 0; other_device_id < thread_info.threads_count; other_device_id++)
    {
      if (other_device_id != thread_info.thread_id)
        {
          int can_access_other_device {};
          throw_on_error (cudaDeviceCanAccessPeer (&can_access_other_device, thread_info.thread_id, other_device_id), __FILE__, __LINE__);

          if (can_access_other_device)
            {
              throw_on_error (cudaDeviceEnablePeerAccess (other_device_id, 0), __FILE__, __LINE__);
            }
          else
            {
              std::cerr << "Warning in thread " << thread_info.thread_id << ": device " << thread_info.thread_id
                        << " can't access device " << other_device_id << " memory!"
                        << " Fall back to normal copy through the host." << std::endl;
              nvlink_enabled = false;
            }
        }
    }

  for (int tid = 0; tid < thread_info.threads_count; tid++)
    {
      if (tid == thread_info.thread_id)
        {
          if (nvlink_enabled)
            std::cout << "NVLINK enabled on thread " << thread_info.thread_id << std::endl;
        }
      thread_info.sync ();
    }
}

template<typename T>
double elements_to_gb (size_t elements_count)
{
  return static_cast<double> (elements_count) * sizeof (T) / 1024 / 1024 / 1024;
}

void print_memory_info (const grid_info_class &grid_info, const thread_info_class &thread_info)
{
  for (int tid = 0; tid < thread_info.threads_count; tid++)
    {
      if (tid == thread_info.thread_id)
        {
          const size_t elements_to_allocate = grid_info.get_ny () * grid_info.get_nx () * static_cast<int> (fdtd_fields::fields_count);
          std::cout << "Allocating " << elements_to_gb<float> (elements_to_allocate) << " GB "
                    << "on GPU " << thread_info.thread_id << std::endl;
        }
      thread_info.sync ();
    }
}

int main ()
{
  int gpus_count {};
  cudaGetDeviceCount (&gpus_count);

  const float x_size_multiplier = 1.1;
  const float height = 160.0;
  const float width = x_size_multiplier * height;
  const int write_each = 100;
  const int grid_size = 1400;
  const int steps_count = 2200;
  const int process_nx = x_size_multiplier * static_cast<float> (grid_size);
  const int process_ny = grid_size;

  double single_gpu_time {};

  const int source_x_offset = 0;

  for (int devices_count = write_each > 0 ? gpus_count : 1; devices_count <= gpus_count; devices_count++)
    {
      std::cout << "\nStarting measurement for " << devices_count << std::endl;

      std::vector<std::thread> threads;
      threads.reserve (devices_count);

      /// Shared objects
      barrier_class barrier (devices_count);
      grid_barrier_class grid_barrier (devices_count, process_nx, process_ny);
      std::vector<double> elapsed_times (devices_count);

      for (int device_id = 0; device_id < devices_count; device_id++)
        {
          thread_info_class thread_info (device_id, devices_count, barrier);
          threads.emplace_back([thread_info, process_nx, process_ny, steps_count, width, height, write_each, &elapsed_times, &grid_barrier] () {
            try {
              prepare_nvlink (thread_info);

              grid_info_class grid_info (width, height, process_nx, process_ny, thread_info);
              print_memory_info (grid_info, thread_info);
              grid_barrier_accessor_class grid_barrier_accessor = grid_barrier.create_accessor (
                thread_info.thread_id, grid_info.get_nx (), grid_info.get_ny (), static_cast<int> (fdtd_fields::fields_count));

              if (thread_info.threads_count == 1)
                {
                  run_fdtd (steps_count, write_each, source_x_offset, elapsed_times.data (), grid_info, grid_barrier_accessor, thread_info);
                }
              else
                {
                  // run_fdtd (steps_count, write_each, elapsed_times.data (), grid_info, grid_barrier_accessor, thread_info);
                  run_fdtd_copy_overlap (steps_count, write_each, source_x_offset, elapsed_times.data (), grid_info, grid_barrier_accessor, thread_info);
                }
            }
            catch (std::runtime_error &error) {
              std::cerr << "Error in thread " << thread_info.thread_id << ": " << error.what() << std::endl;
            }
          });
        }

      for (auto &thread: threads)
        thread.join ();

      if (devices_count == 1)
        single_gpu_time = elapsed_times[0];

      const double max_time = *std::max_element (elapsed_times.begin (), elapsed_times.end ());
      std::cout << "Parallel efficiency: " << single_gpu_time / max_time / devices_count << " (" << max_time << " s)" << std::endl;
    }

  return 0;
}
