//
// Created by egi on 12/30/19.
//

#ifndef MULTIGPUFDTD_GRID_INFO_H
#define MULTIGPUFDTD_GRID_INFO_H

class grid_info_class
{
public:
  grid_info_class () = delete;
  grid_info_class (int process_nx, int process_ny, const thread_info_class &thread_info)
    : nx (process_nx)
  {
    const int chunk_size = process_ny / thread_info.threads_count;
    ny = thread_info.thread_id == thread_info.threads_count - 1
       ? chunk_size + process_ny % chunk_size
       : chunk_size;
    row_begin_in_process = chunk_size * thread_info.thread_id;
    row_end_in_process = thread_info.thread_id < thread_info.threads_count - 1
                       ? chunk_size * (thread_info.thread_id + 1)
                       : process_ny;
  }

private:
  int nx {};
  int ny {};

  int row_begin_in_process {};
  int row_end_in_process {};
};

#endif //MULTIGPUFDTD_GRID_INFO_H
