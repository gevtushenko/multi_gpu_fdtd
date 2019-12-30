//
// Created by egi on 12/30/19.
//

#ifndef MULTIGPUFDTD_GPU_UTILS_H
#define MULTIGPUFDTD_GPU_UTILS_H

#include <sstream>
#include <iostream>
#include <cuda_runtime.h>

inline void throw_on_error (cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess)
    {
      std::stringstream ss;
      ss << "cuda error in " << file << ":" << line << ": " << cudaGetErrorString (code);
      std::string file_and_line;
      ss >> file_and_line;
      throw std::runtime_error (file_and_line);
    }
}


#endif //MULTIGPUFDTD_GPU_UTILS_H
