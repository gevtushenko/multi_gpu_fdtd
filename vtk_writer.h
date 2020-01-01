//
// Created by egi on 1/1/20.
//

#ifndef MULTIGPUFDTD_VTK_WRITER_H
#define MULTIGPUFDTD_VTK_WRITER_H

#include <string>

void write_vtu (
  const std::string &filename,
  float dx,
  float dy,
  int nx,
  int ny,
  const float *e);

#endif //MULTIGPUFDTD_VTK_WRITER_H
