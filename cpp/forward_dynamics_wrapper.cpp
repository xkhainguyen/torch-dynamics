#include <torch/extension.h>
#include "forward_dynamics.h"


// Cleaner wrapper for the generated dynamics
void _forward_dynamics(double* q_in, double* qdot_in, double* tau_in, double* qddot_out) {
    const double* args[3] = {q_in, qdot_in, tau_in};
    double* res[1] = {qddot_out};
    long long int iw[0];
    double w[0];
    eval_forward_dynamics(args, res, iw, w, 0);
}


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
    _forward_dynamics(q_in_ptr, qdot_in_ptr, tau_in_ptr, qddot_out.data_ptr<double>());

    return qddot_out;
}

// Define the Python module and bindings using Pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_dynamics", &forward_dynamics, "Forward dynamics function");
}