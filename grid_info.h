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
    int devices_count_arg,
    float **grid_arg,
    int nx_arg, int ny_arg,
    int elements_per_cell_arg)
    : own_device_id (device_id_arg)
    , devices_count (devices_count_arg)
    , nx (nx_arg)
    , ny (ny_arg)
    , push_top (push_top_arg)
    , push_bottom (push_bottom_arg)
    , grid (grid_arg)
    , elements_per_cell (elements_per_cell_arg)
  {
    throw_on_error (cudaMalloc (grid + own_device_id, nx * ny * elements_per_cell * sizeof (float)), __FILE__, __LINE__);

    for (int iteration: {0, 1})
      {
        throw_on_error (cudaEventCreateWithFlags (get_top_done (iteration, own_device_id), cudaEventDisableTiming), __FILE__, __LINE__);
        throw_on_error (cudaEventCreateWithFlags (get_bottom_done (iteration, own_device_id), cudaEventDisableTiming), __FILE__, __LINE__);
      }
  }

  ~grid_barrier_accessor_class ()
  {
    try {
      throw_on_error (cudaFree (grid[own_device_id]), __FILE__, __LINE__);

      for (int iteration: {0, 1})
        {
          throw_on_error (cudaEventDestroy (*get_top_done (iteration, own_device_id)), __FILE__, __LINE__);
          throw_on_error (cudaEventDestroy (*get_bottom_done (iteration, own_device_id)), __FILE__, __LINE__);
        }
    } catch (...) {
      std::cerr << "Error in barrier accessor destructor!" << std::endl;
    }
  }

  int get_neighbor_device_top ()
  {
    if (own_device_id == devices_count - 1)
      return 0;
    return own_device_id + 1;
  }

  int get_neighbor_device_bottom ()
  {
    if (own_device_id == 0)
      return devices_count - 1;
    return own_device_id - 1;
  }

  template <typename enum_type>
  float *get_own_data (enum_type field_num)
  {
    return grid[own_device_id] + static_cast<int> (field_num) * (nx * ny) + nx; ///< Skip ghost cells
  }

  template <typename enum_type>
  float *get_top_copy_dst (enum_type field_num)
  {
    return grid[get_neighbor_device_top ()] + static_cast<int> (field_num) * (nx * ny);
  }

  template <typename enum_type>
  float *get_top_copy_src (enum_type field_num)
  {
    return get_own_data (field_num) - 2 * nx + (nx * ny);
  }

  template <typename enum_type>
  float *get_bottom_copy_dst (enum_type field_num)
  {
    return grid[get_neighbor_device_bottom ()] + static_cast<int> (field_num) * (nx * ny) + (nx * ny) - nx;
  }

  template <typename enum_type>
  float *get_bottom_copy_src (enum_type field_num)
  {
    return get_own_data (field_num);
  }

  template <typename enum_type>
  void sync_send (enum_type field_num)
  {
    throw_on_error (cudaMemcpy (get_top_copy_dst (field_num), get_top_copy_src (field_num), nx * sizeof (float), cudaMemcpyDefault), __FILE__, __LINE__);
    throw_on_error (cudaMemcpy (get_bottom_copy_dst (field_num), get_bottom_copy_src (field_num), nx * sizeof (float), cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
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
  int own_device_id {};
  int devices_count {};
  int nx {};
  int ny {};
  cudaEvent_t *push_top {};
  cudaEvent_t *push_bottom {};
  float **grid {};
  const int elements_per_cell {};
};

class grid_barrier_class
{
public:
  explicit grid_barrier_class (int devices_count_arg)
    : devices_count (devices_count_arg)
    , push_top_done (new cudaEvent_t[2 * devices_count])
    , push_bottom_done (new cudaEvent_t[2 * devices_count])
    , grid (new float*[devices_count])
  {
  }

  grid_barrier_accessor_class create_accessor (int device_id, int nx, int ny, int elements_per_cell)
  {
    return {
      push_top_done.get (),
      push_bottom_done.get (),
      device_id,
      devices_count,
      grid.get (),
      nx, ny,
      elements_per_cell
    };
  }

private:
  int devices_count {};
  std::unique_ptr<cudaEvent_t[]> push_top_done;
  std::unique_ptr<cudaEvent_t[]> push_bottom_done;

  std::unique_ptr<float*[]> grid;
};

class grid_info_class
{
public:
  grid_info_class () = delete;
  grid_info_class (
    float width_arg,
    float height_arg,
    int process_nx,
    int process_ny,
    const thread_info_class &thread_info)
    : own_nx (process_nx)
    , width (width_arg)
    , height (height_arg)
    , dx (width / static_cast<float> (process_nx))
    , dy (height / static_cast<float> (process_ny))
  {
    const int chunk_size = process_ny / thread_info.threads_count;
    own_ny = thread_info.thread_id == thread_info.threads_count - 1
       ? chunk_size + process_ny % chunk_size
       : chunk_size;
    row_begin_in_process = chunk_size * thread_info.thread_id;
    row_end_in_process = thread_info.thread_id < thread_info.threads_count - 1
                       ? chunk_size * (thread_info.thread_id + 1)
                       : process_ny;

    nx = own_nx;
    ny = own_ny + 2;
  }

  int get_nx () const { return nx; }
  int get_ny () const { return ny; }

  float get_dx () const { return dx; }
  float get_dy () const { return dy; }

  int get_own_cells_count () const { return own_nx * own_ny; }

  int get_row_begin_in_process () const { return row_begin_in_process; }

private:
  int own_nx {};
  int own_ny {};

  float width = 10.0;
  float height = 10.0;

  float dx {};
  float dy {};

  int nx {};
  int ny {};

  int row_begin_in_process {};
  int row_end_in_process {};
};

#endif //MULTIGPUFDTD_GRID_INFO_H
