#include "fdtd.h"
#include <vector>

constexpr float C0 = 299792458.0f; /// Speed of light [metres per second]

float calculate_dt (float dx, float dy)
{
  const float cfl = 0.3;
  return cfl * std::min (dx, dy) / C0;
}

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
      const float object_y = soil_y - 8.0;
      const float object_size = 3.0;

      if (y < soil_y)
        {
          const float middle_x = static_cast<float> (nx) * dx / 2;
          const float object_x = middle_x;

          // square
          // if (x > middle_x - object_size / 2.0f && x < middle_x + object_size / 2 && y > object_y - object_size / 2.0 && y < object_y + object_size / 2.0)

          // circle
          if ((x - object_x) * (x - object_x) + (y - object_y) * (y - object_y) <= object_size * object_size)
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
  const int cell_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (cell_id < n_cells)
    update_h (nx, cell_id, dx, dy, ez, mh, hx, hy);
}

__global__ void update_h_border_kernel (
  int nx,
  int n_own_y,

  float dx,
  float dy,
  const float * __restrict__ ez,
  const float * __restrict__ mh,
  float * __restrict__ hx,
  float * __restrict__ hy)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < nx)
    update_h (nx, nx * (n_own_y - 1) + tid, dx, dy, ez, mh, hx, hy);
}

__global__ void update_h_bulk_kernel (
  int nx,
  int n_own_y,

  float dx,
  float dy,
  const float * __restrict__ ez,
  const float * __restrict__ mh,
  float * __restrict__ hx,
  float * __restrict__ hy)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < nx * (n_own_y - 1))
    update_h (nx, tid, dx, dy, ez, mh, hx, hy);
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

__global__ void update_e_kernel (
  int nx,
  int n_cells,
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
  const unsigned int cell_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (cell_id < n_cells)
    update_e (nx, cell_id, own_in_process_begin, source_position, t, dx, dy, C0_p_dt, ez, dz, er, hx, hy);
}

__global__ void update_e_bulk_kernel (
  int nx,
  int n_cells,
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
  const unsigned int cell_id = nx + blockIdx.x * blockDim.x + threadIdx.x;

  if (cell_id < n_cells)
    update_e (nx, cell_id, own_in_process_begin, source_position, t, dx, dy, C0_p_dt, ez, dz, er, hx, hy);
}

__global__ void update_e_border_kernel (
  int nx,
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
  const unsigned int cell_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (cell_id < nx)
    update_e (nx, cell_id, own_in_process_begin, source_position, t, dx, dy, C0_p_dt, ez, dz, er, hx, hy);
}

#include "vtk_writer.h"

void write_vtk (
  const std::string &filename,
  const float dx,
  const float dy,
  const unsigned int nx,
  const unsigned int ny,
  const float *e)
{
  FILE * f = fopen (filename.c_str (), "w");

  fprintf (f, "# vtk DataFile Version 3.0\n");
  fprintf (f, "vtk output\n");
  fprintf (f, "ASCII\n");
  fprintf (f, "DATASET UNSTRUCTURED_GRID\n");
  fprintf (f, "POINTS %u double\n", nx * ny * 4);

  for (unsigned int j = 0; j < ny; j++)
    {
      for (unsigned int i = 0; i < nx; i++)
        {
          fprintf (f, "%lf %lf 0.0\n", dx * (i + 0), dy * (j + 0) );
          fprintf (f, "%lf %lf 0.0\n", dx * (i + 1), dy * (j + 0) );
          fprintf (f, "%lf %lf 0.0\n", dx * (i + 1), dy * (j + 1) );
          fprintf (f, "%lf %lf 0.0\n", dx * (i + 0), dy * (j + 1) );
        }
    }

  fprintf (f, "CELLS %u %u\n", nx * ny, nx * ny * 5);

  for (unsigned int j = 0; j < ny; j++)
    {
      for (unsigned int i = 0; i < nx; i++)
        {
          const unsigned int point_offset = (j * nx + i) * 4;
          fprintf (f, "4 %u %u %u %u\n", point_offset + 0, point_offset + 1, point_offset + 2, point_offset + 3);
        }
    }

  fprintf (f, "CELL_TYPES %u\n", nx * ny);
  for (unsigned int i = 0; i < nx * ny; i++)
    fprintf (f, "9\n");

  fprintf (f, "CELL_DATA %u\n", nx * ny);
  fprintf (f, "SCALARS Ez double 1\n");
  fprintf (f, "LOOKUP_TABLE default\n");

  for (unsigned int i = 0; i < nx * ny; i++)
    fprintf (f, "%lf\n", e[i]);

  fclose (f);
}

void run_fdtd (
  int steps,
  int write_each,
  int source_x_offset,
  double *elapsed_times,
  const grid_info_class &grid_info,
  receiver_writer &receiver,
  vtk_writer &writer,
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
    n_own_cells, nx, grid_info.process_ny, grid_info.get_row_begin_in_process (), dt, grid_info.get_dx (), grid_info.get_dy (),
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

  float *cpu_e = grid_accessor.cpu_field;
  const int source_position = (grid_info.process_ny / 2) * nx + 2 * nx / 5 + source_x_offset;

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaEventRecord (start);

  for (int step = 0; step < steps; step++)
    {
      update_h_kernel<<<blocks_count, threads_per_block>>> (
        nx, n_own_cells, dx, dy, own_ez, own_mh, own_hx, own_hy);

      grid_accessor.sync_send_top (fdtd_fields::hx);
      grid_accessor.sync_send_top (fdtd_fields::hy);
      thread_info.sync ();

      update_e_kernel<<<blocks_count, threads_per_block>>> (
        nx, n_own_cells, grid_info.get_row_begin_in_process(), source_position, t, dx, dy,
        C0_p_dt, own_ez, own_dz, own_er, own_hx, own_hy);

      grid_accessor.sync_send_bottom (fdtd_fields::ez);
      thread_info.sync ();

      if (write_each > 0 && step % write_each == 0)
        {
          /// Write results
          cudaMemcpy (
            cpu_e + nx * grid_info.get_row_begin_in_process (),
            own_ez,
            grid_info.get_own_cells_count () * sizeof (float),
            cudaMemcpyDeviceToHost);

          if (thread_info.thread_id == 0)
            {
              std::cout << "Writing results for step " << step;
              std::cout.flush ();

              writer.write_vtu (cpu_e);
              receiver.set_received_value (cpu_e[source_position]);

              std::cout << " completed" << std::endl;
            }
          thread_info.sync ();
        }

      t += dt;
    }

  float milliseconds = 0.0;
  cudaEventRecord (stop);
  cudaEventSynchronize (stop);
  cudaEventElapsedTime (&milliseconds, start, stop);

  cudaEventDestroy (stop);
  cudaEventDestroy (start);

  elapsed_times[thread_info.thread_id] = milliseconds / 1000.0f;

  throw_on_error (cudaDeviceSynchronize (), __FILE__, __LINE__);
}

void run_fdtd_copy_overlap (
  int steps,
  int write_each,
  int source_x_offset,
  double *elapsed_times,
  const grid_info_class &grid_info,
  receiver_writer &receiver,
  vtk_writer &writer,
  grid_barrier_accessor_class &grid_accessor,
  const thread_info_class &thread_info)
{
  const float dt = calculate_dt (grid_info.get_dx (), grid_info.get_dy ());

  constexpr unsigned int threads_per_block = 1024;
  const int n_own_cells = grid_info.get_own_cells_count ();
  const int nx = grid_info.get_nx ();
  const int blocks_count = (n_own_cells + threads_per_block - 1) / threads_per_block;
  const int borders_blocks_count = (nx + threads_per_block - 1) / threads_per_block;
  const int bulk_blocks_count = (n_own_cells - nx + threads_per_block - 1) / threads_per_block;

  float *own_er = grid_accessor.get_own_data (fdtd_fields::er);
  float *own_hr = grid_accessor.get_own_data (fdtd_fields::hr);
  float *own_mh = grid_accessor.get_own_data (fdtd_fields::mh);
  float *own_hx = grid_accessor.get_own_data (fdtd_fields::hx);
  float *own_hy = grid_accessor.get_own_data (fdtd_fields::hy);
  float *own_ez = grid_accessor.get_own_data (fdtd_fields::ez);
  float *own_dz = grid_accessor.get_own_data (fdtd_fields::dz);

  initialize_fields<<<blocks_count, threads_per_block>>> (
    n_own_cells, nx, grid_info.process_ny, grid_info.get_row_begin_in_process (), dt, grid_info.get_dx (), grid_info.get_dy (),
    own_er, own_hr, own_mh, own_hx, own_hy, own_ez, own_dz);

  thread_info.sync (); ///< Wait for other threads to complete allocation
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

  int least_priority {};
  int highest_priority {};
  throw_on_error (cudaDeviceGetStreamPriorityRange (&least_priority, &highest_priority), __FILE__, __LINE__);

  cudaStream_t compute_stream, push_top_stream, push_bottom_stream;
  throw_on_error (cudaStreamCreateWithPriority (&compute_stream, cudaStreamDefault, least_priority), __FILE__, __LINE__);
  throw_on_error (cudaStreamCreateWithPriority (&push_top_stream, cudaStreamDefault, highest_priority), __FILE__, __LINE__);
  throw_on_error (cudaStreamCreateWithPriority (&push_bottom_stream, cudaStreamDefault, least_priority), __FILE__, __LINE__);

  float *cpu_e = grid_accessor.cpu_field;

  const std::vector<fdtd_fields> fields_to_update_after_h = {
    fdtd_fields::hx,
    fdtd_fields::hy,
  };

  const std::vector<fdtd_fields> fields_to_update_after_e = {
    fdtd_fields::ez,
  };

  const int source_position = (grid_info.process_ny / 2) * nx + 2 * nx / 5 + source_x_offset;

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaEventRecord (start);

  for (int step = 0; step < steps; step++)
    {
      /// Compute bulk
      update_h_bulk_kernel<<<bulk_blocks_count, threads_per_block, 0, compute_stream>>> (nx, grid_info.get_n_own_y (), dx, dy, own_ez, own_mh, own_hx, own_hy);

      /// Compute boundaries
      update_h_border_kernel<<<borders_blocks_count, threads_per_block, 0, push_top_stream>>> (nx, grid_info.get_n_own_y (), dx, dy, own_ez, own_mh, own_hx, own_hy);
      grid_accessor.async_send_top (fields_to_update_after_h, push_top_stream);

      cudaStreamSynchronize (push_top_stream);
      cudaStreamSynchronize (compute_stream);
      thread_info.sync ();

      /// Compute bulk
      update_e_bulk_kernel<<<bulk_blocks_count, threads_per_block, 0, compute_stream>>> (
        nx, n_own_cells, grid_info.get_row_begin_in_process(), source_position, t, dx, dy,
          C0_p_dt, own_ez, own_dz, own_er, own_hx, own_hy);

      /// Compute boundaries
      update_e_border_kernel<<<borders_blocks_count, threads_per_block, 0, push_bottom_stream>>> (
        nx, grid_info.get_row_begin_in_process(), source_position, t, dx, dy,
          C0_p_dt, own_ez, own_dz, own_er, own_hx, own_hy);
      grid_accessor.async_send_bottom (fields_to_update_after_e, push_bottom_stream);

      cudaStreamSynchronize (push_bottom_stream);
      cudaStreamSynchronize (compute_stream);
      thread_info.sync ();

      if (write_each > 0 && step % write_each == 0)
        {
          /// Write results
          cudaMemcpy (
            cpu_e + nx * grid_info.get_row_begin_in_process (),
            own_ez,
            grid_info.get_own_cells_count () * sizeof (float),
            cudaMemcpyDeviceToHost);

          thread_info.sync ();
          if (thread_info.thread_id == 0)
            {
              std::cout << "Writing results for step " << step;
              std::cout.flush ();
              writer.write_vtu (cpu_e);
              receiver.set_received_value (cpu_e[source_position]);
              // write_vtk ("out_" + std::to_string (step) + ".vtk", dx, dy, grid_info.process_nx, grid_info.process_ny, cpu_e);
              // write_vtu ("out_" + std::to_string (step) + ".vtr", dx, dy, grid_info.process_nx, grid_info.process_ny, cpu_e);
              std::cout << " completed" << std::endl;
            }
          thread_info.sync ();
        }

      t += dt;
    }

  float milliseconds = 0.0;
  cudaEventRecord (stop);
  cudaEventSynchronize (stop);
  cudaEventElapsedTime (&milliseconds, start, stop);

  cudaEventDestroy (stop);
  cudaEventDestroy (start);

  throw_on_error (cudaStreamDestroy (push_top_stream), __FILE__, __LINE__);
  throw_on_error (cudaStreamDestroy (push_bottom_stream), __FILE__, __LINE__);
  throw_on_error (cudaStreamDestroy (compute_stream), __FILE__, __LINE__);

  elapsed_times[thread_info.thread_id] = milliseconds / 1000.0f;

  throw_on_error (cudaDeviceSynchronize (), __FILE__, __LINE__);
  thread_info.sync ();
}
