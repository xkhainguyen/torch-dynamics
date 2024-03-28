#include "dynamics_cpu.h"

#include <iostream>
#include <generated_dynamics.h>
#include <generated_derivatives.h>


// Cleaner wrapper for the generated dynamics
void _dynamics_cpu(const double* q_in, const double* qdot_in, const double* tau_in, double* qddot_out) {
    const double* arg[3] = {q_in, qdot_in, tau_in};
    double* res[1] = {qddot_out};
    long long int iw[0];
    double w[0];
    eval_forward_dynamics(arg, res, iw, w, 0);
}

void _derivatives_cpu(const double* q_in, const double* qdot_in, const double* tau_in, 
                  double* qddot_jac_qout, double* qddot_jac_qdotout, double* qddot_jac_tauout) {
    const double* arg[3] = {q_in, qdot_in, tau_in};
    double* res[3] = {qddot_jac_qout, qddot_jac_qdotout, qddot_jac_tauout};
    long long int iw[0];
    double w[0];
    eval_forward_derivatives(arg, res, iw, w, 0);
}


// Multi-threaded CPU code
template <typename scalar_t>
void dynamics_kernel_cpu(const scalar_t* q_in_ptr, const scalar_t* qdot_in_ptr, const scalar_t* tau_in_ptr, scalar_t* qddot_out_ptr, int q_size, int batch_size) {
    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t b=start; b<end; b++) {
            _dynamics_cpu(q_in_ptr + b*q_size, qdot_in_ptr + b*q_size, tau_in_ptr + b*q_size, qddot_out_ptr + b*q_size);
        }
    });
}

template <typename scalar_t>
void derivatives_kernel_cpu(const scalar_t* q_in_ptr, const scalar_t* qdot_in_ptr, const scalar_t* tau_in_ptr, scalar_t* qddot_jac_qout_ptr, scalar_t* qddot_jac_qdotout_ptr, scalar_t* qddot_jac_tauout_ptr, int q_size, int batch_size) {
    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t b=start; b<end; b++) {
            _derivatives_cpu(q_in_ptr + b*q_size, qdot_in_ptr + b*q_size, tau_in_ptr + b*q_size, qddot_jac_qout_ptr + b*q_size*q_size, qddot_jac_qdotout_ptr + b*q_size*q_size, qddot_jac_tauout_ptr + b*q_size*q_size);
        }
    });
}


// Torch CPU wrapper
torch::Tensor dynamics_cpu(torch::Tensor q_in, torch::Tensor qdot_in, torch::Tensor tau_in) {
    int batch_size = q_in.size(0);
    int q_size = q_in.size(1);
    torch::Tensor qddot_out = torch::zeros_like(q_in);
    using scalar_t = double; 

    dynamics_kernel_cpu<scalar_t>(
        q_in.data_ptr<scalar_t>(), 
        qdot_in.data_ptr<scalar_t>(), 
        tau_in.data_ptr<scalar_t>(), 
        qddot_out.data_ptr<scalar_t>(), 
        q_size,
        batch_size);

    return qddot_out;
}

// Here just return Jacobians, Jacobian-vector product in PyTorch later
std::vector<torch::Tensor>  derivatives_cpu(torch::Tensor q_in, torch::Tensor qdot_in, torch::Tensor tau_in) {
    int batch_size = q_in.size(0);
    int q_size = q_in.size(1);
    torch::Tensor qddot_jac_qout = torch::zeros({batch_size, q_size, q_size}, q_in.options());
    torch::Tensor qddot_jac_qdotout = torch::zeros({batch_size, q_size, q_size}, q_in.options());
    torch::Tensor qddot_jac_tauout = torch::zeros({batch_size, q_size, q_size}, q_in.options());
    using scalar_t = double; 

    derivatives_kernel_cpu<scalar_t>(
        q_in.data_ptr<scalar_t>(), 
        qdot_in.data_ptr<scalar_t>(), 
        tau_in.data_ptr<scalar_t>(), 
        qddot_jac_qout.data_ptr<scalar_t>(), 
        qddot_jac_qdotout.data_ptr<scalar_t>(),
        qddot_jac_tauout.data_ptr<scalar_t>(),
        q_size,
        batch_size);

    return {qddot_jac_qout, qddot_jac_qdotout, qddot_jac_tauout};
}