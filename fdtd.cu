#include "fdtd.cuh"
#include "fdtd.h"
#include <vector>

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
      const float object_1_y = soil_y - 22.0;
      const float object_2_y = soil_y - 24.0;
      const float object_1_size = 3.0;
      const float object_2_size = 8.0;
      const float soil_er_hr = 1.0; // 1.5

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
            er = hr = soil_er_hr;
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

__global__ void update_h_b2r_border_kernel (
  int R,
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
  const int ghost_layer_size = nx * (1 + R);

  if (tid < ghost_layer_size) {
      update_h (nx, tid, dx, dy, ez, mh, hx, hy); ///< Bottom part
  }
  else if (tid < ghost_layer_size * 2) {
      update_h (nx, nx * (n_own_y - (R + 1)) + tid - ghost_layer_size, dx, dy, ez, mh, hx, hy);
  }
}

__global__ void update_h_b2r_bulk_kernel (
  int R,
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

  if (tid < nx * (n_own_y - 1 - R))
    update_h (nx, tid + nx * (R + 1), dx, dy, ez, mh, hx, hy);
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

__global__ void update_e_b2r_bulk_kernel (
  int R,
  int nx,
  int n_own_y,
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
  const unsigned int cell_id = nx * (R + 1) + blockIdx.x * blockDim.x + threadIdx.x;

  if (cell_id < nx * (n_own_y - 1 - R))
    update_e (nx, cell_id, own_in_process_begin, source_position, t, dx, dy, C0_p_dt, ez, dz, er, hx, hy);
}

__global__ void update_e_b2r_border_kernel (
  int R,
  int nx,
  int n_own_y,
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
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int ghost_layer_size = nx * (1 + R);

  if (tid < ghost_layer_size) {
      update_e (nx, tid, own_in_process_begin, source_position, t, dx, dy, C0_p_dt, ez, dz, er, hx, hy);
    }
  else if (tid < ghost_layer_size * 2) {
      update_e (nx, nx * (n_own_y - (R + 1)) + tid - ghost_layer_size, own_in_process_begin, source_position, t, dx, dy, C0_p_dt, ez, dz, er, hx, hy);
    }
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

  std::cout << "Transfer size for R=0 is " << static_cast<double> (grid_accessor.ghost_layer_size_in_bytes ()) / 1024 << " KB\n";

  float milliseconds = 0.0;
  cudaEventRecord (stop);
  cudaEventSynchronize (stop);
  cudaEventElapsedTime (&milliseconds, start, stop);

  cudaEventDestroy (stop);
  cudaEventDestroy (start);

  elapsed_times[thread_info.thread_id] = milliseconds / 1000.0f;

  throw_on_error (cudaDeviceSynchronize (), __FILE__, __LINE__);
}

void run_fdtd_b2r_copy_overlap (
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

  const int max_R = grid_info.get_R ();
  constexpr unsigned int threads_per_block = 256;
  const int n_own_cells = grid_info.get_own_cells_count ();
  const int nx = grid_info.get_nx ();
  const int blocks_count = (n_own_cells + threads_per_block - 1) / threads_per_block;
  const int borders_blocks_count = (2 * nx * (1 + max_R) + threads_per_block - 1) / threads_per_block;
  const int bulk_blocks_count = (n_own_cells - 2 * (nx * (1 + max_R)) + threads_per_block - 1) / threads_per_block;

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
  throw_on_error (cudaStreamCreateWithPriority (&push_bottom_stream, cudaStreamDefault, highest_priority), __FILE__, __LINE__);

  float *cpu_e = grid_accessor.cpu_field;

  const std::vector<fdtd_fields> fields_to_update_after_h = {
    fdtd_fields::hx,
    fdtd_fields::hy,
  };

  const std::vector<fdtd_fields> fields_to_update_after_e = {
    fdtd_fields::ez,
    fdtd_fields::dz,
  };

  const int source_position = (grid_info.process_ny / 2) * nx + 2 * nx / 5 + source_x_offset;

  /// 907 908
  std::unique_ptr<float[]> gpu_data_copy (new float[nx * grid_info.get_ny ()]);
  vtk_writer gpu_writer (dx, dy, nx, grid_info.get_ny (), "gpu_data_dz_" + std::to_string (thread_info.thread_id));
  const float *array_to_write = own_dz;
  bool debug_writes = false;

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaEventRecord (start);

  if (thread_info.thread_id == 0)
    std::cout << "Transfer size for R=" << max_R << " is " << static_cast<double> (grid_accessor.ghost_layer_size_in_bytes ()) / 1024 << " KB\n";

  for (int step = 0; step < steps; step++)
    {
      if (max_R > 0)
        {
          for (int R = max_R; R > 0; R--)
            {
              if (step >= steps)
                break;

              const int cells_to_process = n_own_cells + nx * R * 2;
              const int R_blocks_count = (cells_to_process + threads_per_block - 1) / threads_per_block;

              update_h_kernel<<<R_blocks_count, threads_per_block, 0, compute_stream>>> (nx, cells_to_process, dx, dy, own_ez - R * nx, own_mh - R * nx, own_hx - R * nx, own_hy - R * nx);
              update_e_kernel<<<R_blocks_count, threads_per_block, 0, compute_stream>>> (
                nx, cells_to_process, grid_info.get_row_begin_in_process() - R, source_position, t, dx, dy,
                  C0_p_dt, own_ez - R * nx, own_dz - R * nx, own_er - R * nx, own_hx - R * nx, own_hy - R * nx);

              if (write_each > 0 && step % write_each == 0)
                {
                  if (debug_writes) {
                      cudaMemcpy (gpu_data_copy.get (), array_to_write - nx * (max_R + 1), (n_own_cells + nx * 2 * (max_R + 1)) * sizeof (float), cudaMemcpyDeviceToHost);
                      gpu_writer.write_vtu (gpu_data_copy.get ());
                  }

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
                      std::cout << " completed (R=" << R << ")" << std::endl;
                    }
                  thread_info.sync ();
                }

              cudaStreamSynchronize (compute_stream);

              t += dt;
              step++;
            }

          cudaDeviceSynchronize(); // TODO Replace with event wait
          thread_info.sync ();
        }

      /// Compute bulk
      update_h_b2r_bulk_kernel<<<bulk_blocks_count, threads_per_block, 0, compute_stream>>> (max_R, nx, grid_info.get_n_own_y (), dx, dy, own_ez, own_mh, own_hx, own_hy);

      /// Compute boundaries
      update_h_b2r_border_kernel<<<borders_blocks_count, threads_per_block, 0, push_top_stream>>> (max_R, nx, grid_info.get_n_own_y (), dx, dy, own_ez, own_mh, own_hx, own_hy);
      grid_accessor.async_send_top (fields_to_update_after_h, push_top_stream);
      grid_accessor.async_send_bottom (fields_to_update_after_h, push_top_stream);

      cudaStreamSynchronize (push_top_stream);
      cudaStreamSynchronize (compute_stream);
      thread_info.sync ();

      /// Compute bulk
      update_e_b2r_bulk_kernel<<<bulk_blocks_count, threads_per_block, 0, compute_stream>>> (
        max_R, nx, grid_info.get_n_own_y (), grid_info.get_row_begin_in_process(), source_position, t, dx, dy,
          C0_p_dt, own_ez, own_dz, own_er, own_hx, own_hy);

      /// Compute boundaries
      update_e_b2r_border_kernel<<<borders_blocks_count, threads_per_block, 0, push_bottom_stream>>> (
        max_R, nx, grid_info.get_n_own_y (), grid_info.get_row_begin_in_process(), source_position, t, dx, dy,
          C0_p_dt, own_ez, own_dz, own_er, own_hx, own_hy);
      grid_accessor.async_send_bottom (fields_to_update_after_e, push_bottom_stream);
      grid_accessor.async_send_top (fields_to_update_after_e, push_bottom_stream);

      cudaStreamSynchronize (push_bottom_stream);
      cudaStreamSynchronize (compute_stream);
      thread_info.sync ();

      if (write_each > 0 && step % write_each == 0)
        {
          if (debug_writes)
            {
              cudaMemcpy (gpu_data_copy.get (), array_to_write - nx * (max_R + 1), (n_own_cells + nx * 2 * (max_R + 1)) * sizeof (float), cudaMemcpyDeviceToHost);
              gpu_writer.write_vtu (gpu_data_copy.get ());
            }

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
  throw_on_error (cudaStreamCreateWithPriority (&push_bottom_stream, cudaStreamDefault, highest_priority), __FILE__, __LINE__);

  float *cpu_e = grid_accessor.cpu_field;

  const std::vector<fdtd_fields> fields_to_update_after_h = {
    fdtd_fields::hx,
    fdtd_fields::hy,
  };

  const std::vector<fdtd_fields> fields_to_update_after_e = {
    fdtd_fields::ez,
  };

  const int source_position = (2 * grid_info.process_ny / 3) * nx + 2 * nx / 5 + source_x_offset;

  cudaEvent_t h_bulk_computed, h_border_computed;
  cudaEvent_t e_bulk_computed, e_border_computed;

  cudaEventCreateWithFlags (&h_bulk_computed, cudaEventDisableTiming);
  cudaEventCreateWithFlags (&h_border_computed, cudaEventDisableTiming);

  cudaEventCreateWithFlags (&e_bulk_computed, cudaEventDisableTiming);
  cudaEventCreateWithFlags (&e_border_computed, cudaEventDisableTiming);

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaEventRecord (start);

  for (int step = 0; step < steps; step++)
    {
      if (step % 4 == 0)
        cudaStreamSynchronize (compute_stream);

      /// Compute bulk
      cudaStreamWaitEvent (compute_stream, e_border_computed, 0);
      update_h_bulk_kernel<<<bulk_blocks_count, threads_per_block, 0, compute_stream>>> (nx, grid_info.get_n_own_y (), dx, dy, own_ez, own_mh, own_hx, own_hy);
      cudaEventRecord(h_bulk_computed, compute_stream);

      /// Compute boundaries
      cudaStreamWaitEvent (push_top_stream, e_bulk_computed, 0);
      cudaStreamWaitEvent (push_top_stream, *grid_accessor.get_bottom_done (grid_accessor.get_neighbor_device_top ()), 0);
      update_h_border_kernel<<<borders_blocks_count, threads_per_block, 0, push_top_stream>>> (nx, grid_info.get_n_own_y (), dx, dy, own_ez, own_mh, own_hx, own_hy);
      cudaEventRecord(h_border_computed, push_top_stream);

      grid_accessor.async_send_top (fields_to_update_after_h, push_top_stream);
      thread_info.sync ();

      /// Compute bulk
      cudaStreamWaitEvent (compute_stream, h_border_computed, 0);
      update_e_bulk_kernel<<<bulk_blocks_count, threads_per_block, 0, compute_stream>>> (
        nx, n_own_cells, grid_info.get_row_begin_in_process(), source_position, t, dx, dy,
          C0_p_dt, own_ez, own_dz, own_er, own_hx, own_hy);
      cudaEventRecord(e_bulk_computed, compute_stream);

      /// Compute boundaries
      cudaStreamWaitEvent (push_bottom_stream, h_bulk_computed, 0);
      cudaStreamWaitEvent (push_bottom_stream, *grid_accessor.get_top_done (grid_accessor.get_neighbor_device_bottom ()), 0);
      update_e_border_kernel<<<borders_blocks_count, threads_per_block, 0, push_bottom_stream>>> (
        nx, grid_info.get_row_begin_in_process(), source_position, t, dx, dy,
          C0_p_dt, own_ez, own_dz, own_er, own_hx, own_hy);
      cudaEventRecord(e_border_computed, push_bottom_stream);

      grid_accessor.async_send_bottom (fields_to_update_after_e, push_bottom_stream);
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

  cudaEventDestroy (e_bulk_computed);
  cudaEventDestroy (e_border_computed);
  cudaEventDestroy (h_bulk_computed);
  cudaEventDestroy (h_border_computed);

  throw_on_error (cudaStreamDestroy (push_top_stream), __FILE__, __LINE__);
  throw_on_error (cudaStreamDestroy (push_bottom_stream), __FILE__, __LINE__);
  throw_on_error (cudaStreamDestroy (compute_stream), __FILE__, __LINE__);

  elapsed_times[thread_info.thread_id] = milliseconds / 1000.0f;

  throw_on_error (cudaDeviceSynchronize (), __FILE__, __LINE__);
  thread_info.sync ();
}

