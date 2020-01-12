#ifndef FDTD_CUH_
#define FDTD_CUH_

#include <algorithm>

constexpr float C0 = 299792458.0f; /// Speed of light [metres per second]

float calculate_dt (float dx, float dy)
{
  const float cfl = 0.3;
  return cfl * std::min (dx, dy) / C0;
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
  const int top_neighbor_id = nx * (cell_y + 1) + cell_x;
  return (ez[top_neighbor_id] - ez[cell_id]) / dy;
}

/**
 * @param i Column index
 * @param j Row index
 */
__device__ float update_curl_ey (
  int nx,
  int cell_x,
  int cell_y,
  int cell_id,
  float dx,
  const float * ez)
{
  const int right_neighbor_id = cell_x == nx - 1 ? cell_y * nx + 0 : cell_id + 1;
  return -(ez[right_neighbor_id] - ez[cell_id]) / dx;
}

__device__ void update_h (
  int nx,
  int cell_id,

  float dx,
  float dy,
  const float * __restrict__ ez,
  const float * __restrict__ mh,
  float * __restrict__ hx,
  float * __restrict__ hy)
{
  const int cell_x = cell_id % nx;
  const int cell_y = cell_id / nx;

  const float cex = update_curl_ex (nx, cell_x, cell_y, cell_id, dy, ez);
  const float cey = update_curl_ey (nx, cell_x, cell_y, cell_id, dx, ez);

  // update_h
  hx[cell_id] -= mh[cell_id] * cex;
  hy[cell_id] -= mh[cell_id] * cey;
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
  const int left_neighbor_id = cell_x == 0 ? cell_y * nx + nx - 1 : cell_id - 1;
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

__device__ void update_e (
  int nx,
  int cell_id,
  int own_in_process_begin,
  int source_position,

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
  const int cell_x = cell_id % nx;
  const int cell_y = cell_id / nx;

  const float chz = update_curl_h (nx, cell_id, cell_x, cell_y, dx, dy, hx, hy);
  dz[cell_id] += C0_p_dt * chz;

  if ((own_in_process_begin + cell_y) * nx + cell_x == source_position)
    dz[cell_id] += calculate_source (t, 5E+7);

  ez[cell_id] = dz[cell_id] / er[cell_id];
}

#endif