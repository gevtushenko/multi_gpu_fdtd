#include <thread>
#include <vector>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>

void throw_on_error (cudaError_t code, const char *file, int line)
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

int main ()
{
  int gpus_count {};
  cudaGetDeviceCount (&gpus_count);

  for (int devices_count = 1; devices_count <= gpus_count; devices_count++)
    {
      std::cout << "\nStarting measurement for " << devices_count << std::endl;

      std::vector<std::thread> threads;
      threads.reserve (devices_count);

      for (int device_id = 0; device_id < devices_count; device_id++)
        {
          threads.emplace_back([device_id, devices_count] () {
            try {
                throw_on_error (cudaSetDevice (device_id), __FILE__, __LINE__);

                bool nvlink_enabled = true;

                for (int other_device_id = 0; other_device_id < devices_count; other_device_id++)
                  {
                    if (other_device_id != device_id)
                      {
                        int can_access_other_device {};
                        throw_on_error (cudaDeviceCanAccessPeer (&can_access_other_device, device_id, other_device_id), __FILE__, __LINE__);

                        if (can_access_other_device)
                          {
                            throw_on_error (cudaDeviceEnablePeerAccess (other_device_id, 0), __FILE__, __LINE__);
                          }
                        else
                          {
                            std::cerr << "Warning in thread " << device_id << ": device " << device_id
                                      << " can't access device " << other_device_id << " memory!"
                                      << " Fall back to normal copy through the host." << std::endl;
                            nvlink_enabled = false;
                          }
                      }
                  }

              if (nvlink_enabled)
                std::cout << "NVLINK enabled on thread " << device_id << "\n";
            }
            catch (std::runtime_error &error) {
              std::cerr << "Error in thread " << device_id << ": " << error.what() << std::endl;
            }
          });
        }

      for (auto &thread: threads)
        thread.join ();
    }

  return 0;
}
