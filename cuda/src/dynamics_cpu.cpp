#include "dynamics_cpu.h"

#include <iostream>
#include <generated_dynamics.h>


// Cleaner wrapper for the generated dynamics
void _dynamics(double* q_in, double* qdot_in, double* tau_in, double* qddot_out) {
    const double* args[3] = {q_in, qdot_in, tau_in};
    double* res[1] = {qddot_out};
    long long int iw[0];
    double w[0];
    eval_forward_dynamics(args, res, iw, w, 0);
}

template <typename scalar_t>
void dynamics_kernel((const scalar_t* q_in_ptr, const scalar_t* qdot_in_ptr, const scalar_t* tau_in_ptr, scalar_t* qddot_out_ptr, int batch_size)) {
    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t b=start; b<end; i++) {
            _dynamics(q_in(q_in_ptr + b*q_in.size(1)), qdot_in(qdot_in_ptr + b*qdot_in_ptr.size(1))), tau_in(tau_in_ptr + b*tau_in.size(1)), qddot_out(qddot_out_ptr + b*qddot_out.size(1));
        }
    });
}

torch::Tensor dynamics_cpu(torch::Tensor q_in, torch::Tensor qdot_in, torch::Tensor tau_in) {
    int batch_size = q_in.size(0);
    torch::Tensor qddot_out;

    DISPATCH_GROUP_AND_FLOATING_TYPES(q_in.type(), qdot_in.type(), tau_in.type(), "dynamics_kernel", ([&] {
        qddot_out = torch::zeros({batch_size, q_in.size(1)}, q_in.options());
        dynamics_kernel<scalar_t, scalar_t, scalar_t>(
            q_in.data_ptr<scalar_t>(), 
            qdot_in.data_ptr<scalar_t>(), 
            tau_in.data_ptr<scalar_t>(), 
            qddot_out.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return qddot_out;
}