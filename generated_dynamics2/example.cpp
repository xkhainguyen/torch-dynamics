#include "generated_cont_dynamics.h"
#include "generated_cont_derivatives.h"
#include "generated_dynamics.h"
#include "generated_derivatives.h"


extern "C"
{
    void cont_forward_dynamics(const double *q_in, const double *qdot_in, const double *tau_in, double *qddot_out)
    {
        const double *arg[3] = {q_in, qdot_in, tau_in};
        double *res[1] = {qddot_out};
        long long int iw[0];
        double w[0];
        eval_cont_forward_dynamics(arg, res, iw, w, 0);
    }

    void forward_dynamics(const double *q_in, const double *qdot_in, const double *tau_in, const double *h_in, double *q_out, double *qdot_out)
    {
        const double *arg[4] = {q_in, qdot_in, tau_in, h_in};
        double *res[2] = {q_out, qdot_out};
        long long int iw[0];
        double w[0];
        eval_forward_dynamics(arg, res, iw, w, 0);
    }

    void forward_derivatives(const double *q_in, const double *qdot_in, const double *tau_in, const double *h_in,
                             double *q_jac_qout, double *q_jac_qdotout, double *q_jac_tauout,
                             double *qdot_jac_qout, double *qdot_jac_qdotout, double *qdot_jac_tauout)
    {
        const double *arg[4] = {q_in, qdot_in, tau_in, h_in};
        double *res[6] = {q_jac_qout, q_jac_qdotout, q_jac_tauout, qdot_jac_qout, qdot_jac_qdotout, qdot_jac_tauout};
        long long int iw[0];
        double w[0];
        eval_forward_derivatives(arg, res, iw, w, 0);
    }
}