#include <torch/extension.h>
#include "forward_dynamics.h" // Include the C++ header file

// Define a function that calls the C++ function with PyTorch tensors
torch::Tensor forward_dynamics(torch::Tensor q_in, torch::Tensor qdot_in, torch::Tensor tau_in) {
    // Check tensor shapes and types if needed

    // Extract data pointers
    double* q_in_ptr = q_in.data_ptr<double>();
    double* qdot_in_ptr = qdot_in.data_ptr<double>();
    double* tau_in_ptr = tau_in.data_ptr<double>();

    // Allocate output tensor
    torch::Tensor qddot_out = torch::empty_like(q_in);

    // Call the C++ function
    forward_dynamics(q_in_ptr, qdot_in_ptr, tau_in_ptr, qddot_out.data_ptr<double>());

    return qddot_out;
}

// Define the Python module and bindings using Pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_dynamics", &forward_dynamics, "Forward dynamics function");
}