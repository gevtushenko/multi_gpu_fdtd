//
// Created by egi on 12/30/19.
//

#ifndef MULTIGPUFDTD_FDTD_H
#define MULTIGPUFDTD_FDTD_H

#include "barrier.h"
#include "grid_info.h"
#include "vtk_writer.h"

enum class fdtd_fields : int
{
  er, hr, mh, hx, hy, ez, dz, fields_count
};

void run_fdtd (
  int steps,
  int write_each,
  int source_x_offset,
  double *elapsed_times,
  const grid_info_class &grid_info,
  receiver_writer &receiver,
  vtk_writer &writer,
  grid_barrier_accessor_class &grid_accessor,
  const thread_info_class &thread_info);

void run_fdtd_copy_overlap (
  int steps,
  int write_each,
  int source_x_offset,
  double *elapsed_times,
  const grid_info_class &grid_info,
  receiver_writer &receiver,
  vtk_writer &writer,
  grid_barrier_accessor_class &grid_accessor,
  const thread_info_class &thread_info);

void run_fdtd_copy_overlap_int_fastdiv (
  int steps,
  int write_each,
  int source_x_offset,
  double *elapsed_times,
  const grid_info_class &grid_info,
  receiver_writer &receiver,
  vtk_writer &writer,
  grid_barrier_accessor_class &grid_accessor,
  const thread_info_class &thread_info);

#endif //MULTIGPUFDTD_FDTD_H
