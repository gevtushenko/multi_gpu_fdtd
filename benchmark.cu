#include "fdtd.cuh"
#include "cuda_benchmark.h"
#include <cuda_runtime.h>

__global__ void initialize_fields (
  int n_cells,
  int nx,
  int ny,
  int own_in_process_y_begin,
  float dt,
  float dx,
  float dy,
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
      float er = 1.0;
      float hr = 1.0;

      const int yi = own_in_process_y_begin + own_cell_id / nx;
      const int xi = own_cell_id % nx;

      const float x = static_cast<float> (xi) * dx;
      const float y = static_cast<float> (yi) * dy;

      const float soil_y = static_cast<float> (ny) * dy / 2.2;
      const float object_1_y = soil_y - 8.0;
      const float object_2_y = soil_y - 18.0;
      const float object_1_size = 3.0;
      const float object_2_size = 8.0;

      if (y < soil_y)
        {
          const float middle_x = static_cast<float> (nx) * dx / 2;
          const float object_1_x = middle_x;
          const float object_2_x = middle_x - 20;

          // square
          // if (x > middle_x - object_size / 2.0f && x < middle_x + object_size / 2 && y > object_y - object_size / 2.0 && y < object_y + object_size / 2.0)

          // circle
          if ((x - object_1_x) * (x - object_1_x) + (y - object_1_y) * (y - object_1_y) <= object_1_size * object_1_size)
            er = hr = 200000; /// Relative permeabuliti of Iron
          else if ((x - object_2_x) * (x - object_2_x) + (y - object_2_y) * (y - object_2_y) <= object_2_size * object_2_size)
            er = hr = 200000; /// Relative permeabuliti of Iron
          else
            er = hr = 1.5;
        }

      own_er[own_cell_id] = er;
      own_hr[own_cell_id] = hr;

      own_hx[own_cell_id] = 0.0;
      own_hy[own_cell_id] = 0.0;
      own_ez[own_cell_id] = 0.0;
      own_dz[own_cell_id] = 0.0;

      own_mh[own_cell_id] = C0 * dt / own_hr[own_cell_id];
    }
}

int main ()
{
  cudaSetDevice (1);

  /// Grid size equal to block size (1024)
  const int nx = 32;
  const int ny = 32;
  const int n_cells = nx * ny;
  const int n_actual_cells = nx * (2 + ny);

  const float dt = 1e-6;
  const float dx = 1e-1;
  const float dy = 1e-1;

  float *actual_er {};
  float *actual_hr {};
  float *actual_mh {};
  float *actual_hx {};
  float *actual_hy {};
  float *actual_ez {};
  float *actual_dz {};

  cudaMalloc (&actual_er, n_actual_cells * sizeof (float));
  cudaMalloc (&actual_hr, n_actual_cells * sizeof (float));
  cudaMalloc (&actual_mh, n_actual_cells * sizeof (float));
  cudaMalloc (&actual_hx, n_actual_cells * sizeof (float));
  cudaMalloc (&actual_hy, n_actual_cells * sizeof (float));
  cudaMalloc (&actual_ez, n_actual_cells * sizeof (float));
  cudaMalloc (&actual_dz, n_actual_cells * sizeof (float));

  float *er = actual_er + nx;
  float *hr = actual_hr + nx;
  float *mh = actual_mh + nx;
  float *hx = actual_hx + nx;
  float *hy = actual_hy + nx;
  float *ez = actual_ez + nx;
  float *dz = actual_dz + nx;

  initialize_fields<<<1, 1024>>> (
    n_actual_cells, nx, ny, 0, dt, dx, dy, er, hr, mh, hx, hy, ez, dz);

  cuda_benchmark::controller controller (1024, 1);

  controller.benchmark ("base h update", [=] __device__ (cuda_benchmark::state &state) {
    const int cell_id = blockIdx.x * blockDim.x + threadIdx.x;

    for (auto _ : state)
      update_h (nx, cell_id, dx, dy, ez, mh, hx, hy);
  });

  controller.benchmark ("shared h update", [=] __device__ (cuda_benchmark::state &state) {
    const int cell_id = threadIdx.x;

    __shared__ float cache[1024 + 32 * 2];

    for (auto _ : state)
      {
        cache[cell_id] = ez[cell_id];
        __syncthreads ();

        update_h (nx, cell_id, dx, dy, cache + 32, mh, hx, hy);
      }
  });

  cudaFree (actual_er);
  cudaFree (actual_hr);
  cudaFree (actual_mh);
  cudaFree (actual_hx);
  cudaFree (actual_hy);
  cudaFree (actual_ez);
  cudaFree (actual_dz);

  return 0;
}