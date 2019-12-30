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

/**
 * Calculate curl of Ex with periodic boundary condition
 * @param i Column index
 * @param j Row index
 */
__device__ float update_curl_ex (
  int nx,
  int cell_x,
  int cell_y,
  int cell_id,
  float dy,
  const float * ez)
{
  const int neighbor_id = nx * (cell_y + 1) + cell_x;
  return (ez[neighbor_id] - ez[cell_id]) / dy;
}

/**
 * @param i Column index
 * @param j Row index
 */
__device__ float update_curl_ey (
  int nx,
  int cell_x,
  int cell_id,
  float dx,
  const float * ez)
{
  const int neighbor_id = cell_x == nx - 1 ? 0 : cell_id + 1;
  return -(ez[neighbor_id] - ez[cell_id]) / dx;
}

__global__ void update_h_kernel (
    int nx,
    int n_cells,

    float dx,
    float dy,
    const float * __restrict__ ez,
    const float * __restrict__ mh,
    float * __restrict__ hx,
    float * __restrict__ hy)
{
  const unsigned int cell_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (cell_id < n_cells)
    {
      const int cell_x = cell_id % nx;
      const int cell_y = cell_id / nx;

      const float cex = update_curl_ex (nx, cell_x, cell_y, cell_id, dy, ez);
      const float cey = update_curl_ey (nx, cell_x, cell_id, dx, ez);

      // update_h
      hx[cell_id] -= mh[cell_id] * cex;
      hy[cell_id] -= mh[cell_id] * cey;
    }
}

__device__ static float update_curl_h (
  int nx,
  int cell_id,
  int cell_x,
  int cell_y,
  float dx,
  float dy,
  const float * __restrict__ hx,
  const float * __restrict__ hy)
{
  const int left_neighbor_id = cell_x == 0 ? nx - 1 : cell_x - 1;
  const int bottom_neighbor_id = nx * (cell_y - 1) + cell_x;

  return (hy[cell_id] - hy[left_neighbor_id]) / dx
       - (hx[cell_id] - hx[bottom_neighbor_id]) / dy;
}

__device__ float gaussian_pulse (float t, float t_0, float tau)
{
  return __expf (-(((t - t_0) / tau) * (t - t_0) / tau));
}

__device__ float calculate_source (float t, float frequency)
{
  const float tau = 0.5f / frequency;
  const float t_0 = 6.0f * tau;
  return gaussian_pulse (t, t_0, tau);
}

__global__ void update_e_kernel (
  int nx,
  int n_cells,
  int own_in_process_begin,

  float t,
  float dx,
  float dy,
  float C0_p_dt,
  float * __restrict__ ez,
  float * __restrict__ dz,
  const float * __restrict__ er,
  const float * __restrict__ hx,
  const float * __restrict__ hy)
{
  const unsigned int cell_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (cell_id < n_cells)
    {
      const int cell_x = cell_id % nx;
      const int cell_y = cell_id / nx;

      const float chz = update_curl_h (nx, cell_id, cell_x, cell_y, dx, dy, hx, hy);
      dz[cell_id] += C0_p_dt * chz;

      if (cell_id == (own_in_process_begin + cell_x - 1) / 2)
        dz[cell_id] += calculate_source (t, 1E+9);

      ez[cell_id] = dz[cell_id] / er[cell_id];
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
  const int nx = grid_info.get_nx ();
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

  float t {};
  const float dx = grid_info.get_dx ();
  const float dy = grid_info.get_dy ();
  const float C0_p_dt = C0 * dt;

  for (int step = 0; step < steps; step++)
    {
      update_h_kernel<<<blocks_count, threads_per_block>>> (
        nx, n_own_cells, dx, dy, own_ez, own_mh, own_hx, own_hy);
      cudaDeviceSynchronize ();
      thread_info.sync ();

      grid_accessor.sync_send (fdtd_fields::hx);
      grid_accessor.sync_send (fdtd_fields::hy);

      update_e_kernel<<<blocks_count, threads_per_block>>> (
        nx, n_own_cells, grid_info.get_row_begin_in_process(), t, dx, dy,
        C0_p_dt, own_ez, own_dz, own_er, own_hx, own_hy);

      grid_accessor.sync_send (fdtd_fields::ez);
      thread_info.sync ();
    }

  throw_on_error (cudaDeviceSynchronize (), __FILE__, __LINE__);
}
