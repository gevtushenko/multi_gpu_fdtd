//
// Created by egi on 1/1/20.
//

#ifndef MULTIGPUFDTD_VTK_WRITER_H
#define MULTIGPUFDTD_VTK_WRITER_H

#include <string>
#include <memory>

struct vtk_writer_impl;

class vtk_writer
{
public:
  vtk_writer (
    float dx_arg,
    float dy_arg,
    int nx_arg,
    int ny_arg,
    const std::string &filename_arg);

  ~vtk_writer ();

  void write_vtu (const float *e);

private:
  float dx, dy;
  int nx {}, ny {};
  std::string filename;
  unsigned int step {};

  std::unique_ptr<vtk_writer_impl> p_impl;
};

class receiver_writer
{
public:
  receiver_writer (
    int time_steps_count_arg,
    int samples_count_arg);

  void set_received_value (float value);

private:
  int time_steps_count {};
  int samples_count {};

  int time {};
  int sample {};

  std::unique_ptr<float[]> values;
  std::unique_ptr<vtk_writer> writer;
};

#endif //MULTIGPUFDTD_VTK_WRITER_H
