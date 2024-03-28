
#ifndef DYNAMICS_CPU_H_
#define DYNAMICS_CPU_H_

#include <torch/extension.h>
#include <vector>


torch::Tensor dynamics_cpu(torch::Tensor, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> derivatives_cpu(torch::Tensor, torch::Tensor, torch::Tensor);


#endif


  