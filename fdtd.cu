#include "fdtd.h"

constexpr float C0 = 299792458.0f; /// Speed of light [metres per second]

float calculate_dt (float dx, float dy)
{
  const float cfl = 0.4;
  return cfl * std::min (dx, dy) / C0;
}

__global__ void initialize_fields (
  int n_cells,
  float dt,
  float *own_er,
  float *own_hr,
  float *own_mh,
  float *own_hx,
  float *own_hy,
  float *own_ez,
  float *own_dz
  )
{
  const unsigned int own_cell_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (own_cell_id < n_cells)
    {
      own_er[own_cell_id] = 1.0;
      own_hr[own_cell_id] = 1.0;

      own_hx[own_cell_id] = 0.0;
      own_hy[own_cell_id] = 0.0;
      own_ez[own_cell_id] = 0.0;
      own_dz[own_cell_id] = 0.0;

      own_mh[own_cell_id] = C0 * dt / own_hr[own_cell_id];
    }
}

void run_fdtd (
  int steps,
  const grid_info_class &grid_info,
  grid_barrier_accessor_class &grid_accessor,
  const thread_info_class &thread_info)
{
  const float dt = calculate_dt (grid_info.get_dx (), grid_info.get_dy ());

  constexpr unsigned int threads_per_block = 1024;
  const int n_own_cells = grid_info.get_own_cells_count ();
  const unsigned int blocks_count = (n_own_cells + threads_per_block - 1) / threads_per_block;

  float *own_er = grid_accessor.get_own_data (fdtd_fields::er);
  float *own_hr = grid_accessor.get_own_data (fdtd_fields::hr);
  float *own_mh = grid_accessor.get_own_data (fdtd_fields::mh);
  float *own_hx = grid_accessor.get_own_data (fdtd_fields::hx);
  float *own_hy = grid_accessor.get_own_data (fdtd_fields::hy);
  float *own_ez = grid_accessor.get_own_data (fdtd_fields::ez);
  float *own_dz = grid_accessor.get_own_data (fdtd_fields::dz);

  initialize_fields<<<blocks_count, threads_per_block>>> (
    n_own_cells, dt,
    own_er, own_hr, own_mh, own_hx, own_hy, own_ez, own_dz);

  thread_info.sync (); ///< Wait for all threads to allocate their fields
  grid_accessor.sync_send (fdtd_fields::er);
  grid_accessor.sync_send (fdtd_fields::hr);
  grid_accessor.sync_send (fdtd_fields::mh);
  grid_accessor.sync_send (fdtd_fields::hx);
  grid_accessor.sync_send (fdtd_fields::hy);
  grid_accessor.sync_send (fdtd_fields::ez);
  grid_accessor.sync_send (fdtd_fields::dz);
  thread_info.sync ();

  throw_on_error (cudaDeviceSynchronize (), __FILE__, __LINE__);
}
