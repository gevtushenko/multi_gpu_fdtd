cmake_minimum_required(VERSION 3.15)
project(MultiGPUFDTD)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -gencode arch=compute_75,code=sm_75")

if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g")
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math -g")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

find_package(VTK COMPONENTS
        vtkCommonCore
        vtkCommonDataModel
        vtkFiltersGeneral
        vtkIOXML
        vtkInteractionStyle
        vtkRenderingCore
        vtkRenderingFreeType
        vtkRenderingOpenGL2 REQUIRED)

cuda_add_executable(MultiGPUFDTD main.cpp barrier.h grid_info.h gpu_utils.h fdtd.cu fdtd.h vtk_writer.h vtk_writer.cpp)
target_include_directories(MultiGPUFDTD PRIVATE ${CUDA_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS})
target_link_libraries(MultiGPUFDTD ${VTK_LIBRARIES})