#include "forward_dynamics.h"

extern "C" {
    void forward_dynamics(double* q_in, double* qdot_in, double* tau_in, double* qddot_out) {
        const double* args[3] = {q_in, qdot_in, tau_in};
        double* res[1] = {qddot_out};
        long long int iw[0];
        double w[0];
        eval_forward_dynamics(args, res, iw, w, 0);
    }
}