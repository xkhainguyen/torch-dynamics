
#ifndef DYNAMICS_GPU_H_
#define DYNAMICS_GPU_H_

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


torch::Tensor dynamics_gpu(torch::Tensor, torch::Tensor, torch::Tensor);
// std::vector<torch::Tensor> derivatives_gpu(int, torch::Tensor, torch::Tensor);

#endif


  