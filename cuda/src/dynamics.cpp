#include <torch/extension.h>
#include "dynamics_gpu.h"
#include "dynamics_cpu.h"


#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


/* Interface for cuda and c++ robot dynamics

*/

torch::Tensor dynamics(torch::Tensor q_in, torch::Tensor qdot_in, torch::Tensor tau_in) {
    CHECK_CONTIGUOUS(q_in);
    CHECK_CONTIGUOUS(qdot_in);
    CHECK_CONTIGUOUS(tau_in);

    if (q_in.device().type() == torch::DeviceType::CPU) {
        return dynamics_cpu(q_in, qdot_in, tau_in);

    } else if (q_in.device().type() == torch::DeviceType::CUDA) {
        return dynamics_gpu(q_in, qdot_in, tau_in);
    }

    return q_in;
}

// // Define a function that calls the C++ function with PyTorch tensors
// torch::Tensor forward_dynamics(torch::Tensor q_in, torch::Tensor qdot_in, torch::Tensor tau_in) {
//     // Check tensor shapes and types if needed

//     // Extract data pointers
//     double* q_in_ptr = q_in.data_ptr<double>();
//     double* qdot_in_ptr = qdot_in.data_ptr<double>();
//     double* tau_in_ptr = tau_in.data_ptr<double>();

//     // Allocate output tensor
//     torch::Tensor qddot_out = torch::empty_like(q_in);

//     // Call the C++ function
//     _forward_dynamics(q_in_ptr, qdot_in_ptr, tau_in_ptr, qddot_out.data_ptr<double>());

//     return qddot_out;
// }

// Define the Python module and bindings using Pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dynamics", &dynamics, "Forward dynamics function");
}