/* This file was automatically generated by CasADi 3.6.3.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

int eval_forward_dynamics(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int eval_forward_dynamics_alloc_mem(void);
int eval_forward_dynamics_init_mem(int mem);
void eval_forward_dynamics_free_mem(int mem);
int eval_forward_dynamics_checkout(void);
void eval_forward_dynamics_release(int mem);
void eval_forward_dynamics_incref(void);
void eval_forward_dynamics_decref(void);
casadi_int eval_forward_dynamics_n_in(void);
casadi_int eval_forward_dynamics_n_out(void);
casadi_real eval_forward_dynamics_default_in(casadi_int i);
const char* eval_forward_dynamics_name_in(casadi_int i);
const char* eval_forward_dynamics_name_out(casadi_int i);
const casadi_int* eval_forward_dynamics_sparsity_in(casadi_int i);
const casadi_int* eval_forward_dynamics_sparsity_out(casadi_int i);
int eval_forward_dynamics_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define eval_forward_dynamics_SZ_ARG 4
#define eval_forward_dynamics_SZ_RES 2
#define eval_forward_dynamics_SZ_IW 0
#define eval_forward_dynamics_SZ_W 53
#ifdef __cplusplus
} /* extern "C" */
#endif
