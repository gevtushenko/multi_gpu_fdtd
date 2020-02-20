//
// Created by egi on 12/30/19.
//

#ifndef MULTIGPUFDTD_GRID_INFO_H
#define MULTIGPUFDTD_GRID_INFO_H

#include <cuda_runtime.h>
#include <vector>
#include <memory>

#include "gpu_utils.h"

class grid_barrier_accessor_class
{
public:
  grid_barrier_accessor_class () = delete;
  grid_barrier_accessor_class (
    int R_arg,
    cudaEvent_t *push_top_arg,
    cudaEvent_t *push_bottom_arg,
    cudaEvent_t *compute_h_arg,
    cudaEvent_t *compute_e_arg,
    int device_id_arg,
    int devices_count_arg,
    float **grid_arg,
    int *nys_arg,
    int nx_arg, int ny_arg,
    int elements_per_cell_arg,
    float *cpu_field_arg)
    : R (R_arg)
    , own_device_id (device_id_arg)
    , devices_count (devices_count_arg)
    , nx (nx_arg)
    , ny (ny_arg)
    , push_top (push_top_arg)
    , push_bottom (push_bottom_arg)
    , compute_h (compute_h_arg)
    , compute_e (compute_e_arg)
    , grid (grid_arg)
    , nys (nys_arg)
    , elements_per_cell (elements_per_cell_arg)
    , cpu_field (cpu_field_arg)
  {
    nys_arg[own_device_id] = ny;
    throw_on_error (cudaMalloc (&grid[own_device_id], nx * ny * elements_per_cell * sizeof (float)), __FILE__, __LINE__);
    throw_on_error (cudaEventCreateWithFlags (get_top_done (own_device_id), cudaEventDisableTiming), __FILE__, __LINE__);
    throw_on_error (cudaEventCreateWithFlags (get_bottom_done (own_device_id), cudaEventDisableTiming), __FILE__, __LINE__);
    throw_on_error (cudaEventCreateWithFlags (get_compute_h_done (own_device_id), cudaEventDisableTiming), __FILE__, __LINE__);
    throw_on_error (cudaEventCreateWithFlags (get_compute_e_done (own_device_id), cudaEventDisableTiming), __FILE__, __LINE__);
  }

  ~grid_barrier_accessor_class ()
  {
    try {
      throw_on_error (cudaFree (grid[own_device_id]), __FILE__, __LINE__);
      throw_on_error (cudaEventDestroy (*get_top_done (own_device_id)), __FILE__, __LINE__);
      throw_on_error (cudaEventDestroy (*get_bottom_done (own_device_id)), __FILE__, __LINE__);
      throw_on_error (cudaEventDestroy (*get_compute_h_done (own_device_id)), __FILE__, __LINE__);
      throw_on_error (cudaEventDestroy (*get_compute_e_done (own_device_id)), __FILE__, __LINE__);
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
    return grid[own_device_id] + static_cast<int> (field_num) * (nx * ny) + ghost_layer_size (); ///< Skip ghost cells
  }

  template <typename enum_type>
  float *get_top_copy_dst (enum_type field_num)
  {
    return grid[get_neighbor_device_top ()] + static_cast<int> (field_num) * (nx * nys[get_neighbor_device_top ()]);
  }

  template <typename enum_type>
  float *get_top_copy_src (enum_type field_num)
  {
    return get_own_data (field_num) - ghost_layer_size () ///< Get into gpu coordinates (not own)
         + (nx * ny)  ///< Past the end
         - 2 * ghost_layer_size (); ///< Step over ghost layer and own data
  }

  template <typename enum_type>
  float *get_bottom_copy_dst (enum_type field_num)
  {
    return grid[get_neighbor_device_bottom ()] + (static_cast<int> (field_num) + 1) * (nx * nys[get_neighbor_device_bottom ()]) - ghost_layer_size ();
  }

  template <typename enum_type>
  float *get_bottom_copy_src (enum_type field_num)
  {
    return get_own_data (field_num);
  }

  template <typename enum_type>
  void sync_send (enum_type field_num)
  {
    sync_send_top (field_num);
    sync_send_bottom (field_num);
  }

  template <typename enum_type>
  void sync_send_top (enum_type field_num)
  {
    throw_on_error (cudaMemcpy (get_top_copy_dst (field_num), get_top_copy_src (field_num), ghost_layer_size_in_bytes(), cudaMemcpyDefault), __FILE__, __LINE__);
  }

  template <typename enum_type>
  void sync_send_bottom (enum_type field_num)
  {
    auto dst = get_bottom_copy_dst (field_num);
    auto src = get_bottom_copy_src (field_num);
    auto n = ghost_layer_size_in_bytes ();
    throw_on_error (cudaMemcpy (dst, src, n, cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
  }

  template <typename enum_type>
  void async_send_top (const std::vector<enum_type> &fields_num, cudaStream_t &stream_top)
  {
    for (auto &field_num: fields_num)
      throw_on_error (cudaMemcpyAsync (get_top_copy_dst (field_num), get_top_copy_src (field_num), ghost_layer_size_in_bytes (), cudaMemcpyDefault, stream_top), __FILE__, __LINE__);
    cudaEventRecord (*get_top_done (own_device_id), stream_top);
  }

  template <typename enum_type>
  void async_send_bottom (const std::vector<enum_type> &fields_num, cudaStream_t &stream_bottom)
  {
    for (auto &field_num: fields_num)
      throw_on_error (cudaMemcpyAsync (get_bottom_copy_dst (field_num), get_bottom_copy_src (field_num), ghost_layer_size_in_bytes (), cudaMemcpyDefault, stream_bottom), __FILE__, __LINE__);
    cudaEventRecord (*get_bottom_done (own_device_id), stream_bottom);
  }

  cudaEvent_t *get_top_done (int device_id)
  {
    return get_event (push_top, device_id);
  }

  cudaEvent_t *get_bottom_done (int device_id)
  {
    return get_event (push_bottom, device_id);
  }

  cudaEvent_t *get_compute_h_done (int device_id)
  {
    return get_event (compute_h, device_id);
  }

  cudaEvent_t *get_compute_e_done (int device_id)
  {
    return get_event (compute_e, device_id);
  }

private:
  cudaEvent_t *get_event (cudaEvent_t *events, int device_id)
  {
    return events + device_id;
  }

  size_t ghost_layer_size () const
  {
    return nx * (R + 1);
  }

  size_t ghost_layer_size_in_bytes () const
  {
    return ghost_layer_size() * sizeof (float);
  }

private:
  int R {};
  int own_device_id {};
  int devices_count {};
  int nx {};
  int ny {};
  cudaEvent_t *push_top {};
  cudaEvent_t *push_bottom {};
  cudaEvent_t *compute_h {};
  cudaEvent_t *compute_e {};
  float **grid {};
  const int *nys {};
  const int elements_per_cell {};

public:
  float *cpu_field {};
};

class grid_barrier_class
{
public:
  explicit grid_barrier_class (int devices_count_arg, int process_nx, int process_ny, int R_arg = 0)
    : devices_count (devices_count_arg)
    , R (R_arg)
    , push_top_done (new cudaEvent_t[devices_count])
    , push_bottom_done (new cudaEvent_t[devices_count])
    , compute_h (new cudaEvent_t[devices_count])
    , compute_e (new cudaEvent_t[devices_count])
    , grid (new float*[devices_count])
    , nys (new int[devices_count])
  {
    throw_on_error (cudaMallocHost (&cpu_field, process_nx * process_ny * sizeof (float)), __FILE__, __LINE__);
  }

  ~grid_barrier_class ()
  {
    cudaFreeHost (cpu_field);
  }

  grid_barrier_accessor_class create_accessor (int device_id, int nx, int ny, int elements_per_cell)
  {
    return {
      R,
      push_top_done.get (),
      push_bottom_done.get (),
      compute_h.get (),
      compute_e.get (),
      device_id,
      devices_count,
      grid.get (), nys.get (),
      nx, ny,
      elements_per_cell,
      cpu_field
    };
  }

private:
  int devices_count {};
  int R {};
  std::unique_ptr<cudaEvent_t[]> push_top_done;
  std::unique_ptr<cudaEvent_t[]> push_bottom_done;
  std::unique_ptr<cudaEvent_t[]> compute_h;
  std::unique_ptr<cudaEvent_t[]> compute_e;

  std::unique_ptr<float*[]> grid;
  std::unique_ptr<int[]> nys;
  float *cpu_field {};
};

class grid_info_class
{
public:
  grid_info_class () = delete;
  grid_info_class (
    int R_arg,
    float width_arg,
    float height_arg,
    int process_nx_arg,
    int process_ny_arg,
    const thread_info_class &thread_info)
    : R (R_arg)
    , process_nx (process_nx_arg)
    , process_ny (process_ny_arg)
    , own_nx (process_nx)
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
    ny = own_ny + 2 * (R + 1);
  }

  int get_R () const { return R; }

  int get_nx () const { return nx; }
  int get_ny () const { return ny; }

  int get_n_own_y () const { return own_ny; }

  float get_dx () const { return dx; }
  float get_dy () const { return dy; }

  int get_own_cells_count () const { return own_nx * own_ny; }

  int get_row_begin_in_process () const { return row_begin_in_process; }

public:
  const int process_nx {};
  const int process_ny {};

private:
  int R {};
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
