//
// Created by egi on 12/30/19.
//

#ifndef MULTIGPUFDTD_FDTD_H
#define MULTIGPUFDTD_FDTD_H

#include "barrier.h"
#include "grid_info.h"

enum class fdtd_fields : int
{
  er, hr, mh, hx, hy, ez, dz, fields_count
};

void run_fdtd (int steps, grid_barrier_accessor_class &grid_accessor, const thread_info_class &thread_info);

#endif //MULTIGPUFDTD_FDTD_H
