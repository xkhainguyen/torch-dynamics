from torch.autograd import Function
import torch
import argparse
import time
import cartpole2l
import ipdb

nq = 3
nqdot = 3
ntau = 3
nqt = nq + nqdot + ntau
        
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=1000)
parser.add_argument('-r', '--runs', type=int, default=1000)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='us')
parser.add_argument('-c', '--cuda', action='store_true', default=False)
parser.add_argument('-d', '--double', action='store_true', default=True)
options = parser.parse_args()
print(options)

device = torch.device("cuda") if options.cuda else torch.device("cpu")
dtype = torch.float64 if options.double else torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': False}

q_in = torch.randn((options.batch_size, nq), **kwargs)
qdot_in = torch.randn((options.batch_size, nqdot), **kwargs)
tau_in = torch.randn((options.batch_size, ntau), **kwargs)
h_in = torch.full((options.batch_size, nq), 0.1, **kwargs)

# q_in = torch.tensor([[1.1, 2, 3.], [1, 2, 3.]], **kwargs)
# qdot_in = torch.tensor([[1, 2, 3.], [1, 2, 3.]], **kwargs)
# tau_in = torch.tensor([[2.0, 0, 1.], [2, 0, 0.]], **kwargs)
eps = 1e-8
q_id = eps*torch.eye(nq, **kwargs)[None].repeat(options.batch_size, 1, 1)
qdot_id = eps*torch.eye(nqdot, **kwargs)[None].repeat(options.batch_size, 1, 1)
tau_id = eps*torch.eye(ntau, **kwargs)[None].repeat(options.batch_size, 1, 1)

@torch.jit.script
def finite_diff_pre_processing(q, qdot, tau, h, q_id, qdot_id, tau_id):
    nq = 3
    nqdot = 3
    ntau = 3
    nqt = nq + nqdot + ntau
        
    ### compute parallel finite differences
    q_plus = q[:,None] + q_id
    q_minus = q[:,None] - q_id
    q_zero = q[:,None].repeat(1, nqdot+ntau, 1)
    q_total = torch.cat((q_minus, q_zero, q_plus, q_zero), dim=1)
    # ipdb.set_trace()
    qdot_plus = qdot[:,None] + qdot_id
    qdot_minus = qdot[:,None] - qdot_id
    qdot_zero_q = qdot[:,None].repeat(1, nq, 1)
    qdot_zero_tau = qdot[:,None].repeat(1, ntau, 1)
    qdot_total = torch.cat((qdot_zero_q, qdot_minus, qdot_zero_tau, qdot_zero_q, qdot_plus, qdot_zero_tau), dim=1)
    tau_plus = tau[:,None] + tau_id
    tau_minus = tau[:,None] - tau_id
    tau_zero = tau[:,None].repeat(1, nq+nqdot, 1)
    tau_total = torch.cat((tau_zero, tau_minus, tau_zero, tau_plus), dim=1)
    h_total = h[:,None].repeat(1, 2*(nq+nqdot+ntau), 1)
    # ipdb.set_trace()
    return q_total.reshape(-1, nq), qdot_total.reshape(-1, nqdot), tau_total.reshape(-1, ntau), h_total.reshape(-1, nqt)

@torch.jit.script
def finite_diff_post_processing(q_out, qdot_out):
    nq = 3
    nqdot = 3
    ntau = 3
    nqt = nq + nqdot + ntau
    eps = 1e-8

    q_out = q_out.reshape(-1, 2*nqt, nq)
    qdot_out = qdot_out.reshape(-1, 2*nqt, nqdot)
        
    q_jac_q = -(q_out[:,:nq] - q_out[:,nqt:nq+nqt]) / (2*eps)
    q_jac_qdot = -(q_out[:,nq:nq+nqdot] - q_out[:,nqt+nq:nqt+nq+nqdot]) / (2*eps)
    q_jac_tau = -(q_out[:,nq+nqdot:nqt] - q_out[:,nqt+nq+nqdot:]) / (2*eps)
    qdot_jac_q = -(qdot_out[:,:nq] - qdot_out[:,nqt:nq+nqt]) / (2*eps) 
    qdot_jac_qdot = -(qdot_out[:,nq:nq+nqdot] - qdot_out[:,nqt+nq:nqt+nq+nqdot]) / (2*eps)
    qdot_jac_tau = -(qdot_out[:,nq+nqdot:nqt] - qdot_out[:,nqt+nq+nqdot:]) / (2*eps)
    # ipdb.set_trace()
    return q_jac_q, q_jac_qdot, q_jac_tau, qdot_jac_q, qdot_jac_qdot, qdot_jac_tau

def finite_diff_derivatives(q, qdot, tau, h):
    q_total, qdot_total, tau_total, h_total = finite_diff_pre_processing(q, qdot, tau, h, q_id, qdot_id, tau_id)
    q_out, qdot_out = cartpole2l.dynamics(q_total, qdot_total, tau_total, h_total)
    q_jac_q, q_jac_qdot, q_jac_tau, qdot_jac_q, qdot_jac_qdot, qdot_jac_tau = finite_diff_post_processing(q_out, qdot_out)
    return q_jac_q, q_jac_qdot, q_jac_tau, qdot_jac_q, qdot_jac_qdot, qdot_jac_tau

# warm start the device
for i in range(10):
    # q_out, qdot_out = cartpole2l.dynamics(q_in, qdot_in, tau_in, h_in)
    q_jac_q, q_jac_qdot, q_jac_tau, qdot_jac_q, qdot_jac_qdot, qdot_jac_tau = cartpole2l.derivatives(q_in, qdot_in, tau_in, h_in)
    q_jac_q_fd, q_jac_qdot_fd, q_jac_tau_fd, qdot_jac_q_fd, qdot_jac_qdot_fd, qdot_jac_tau_fd = finite_diff_derivatives(q_in, qdot_in, tau_in, h_in)
    
    # compare the relative error between the two
    # print("q_jac_error: ", torch.norm(q_jac_q - q_jac_q_fd) / torch.norm(q_jac_q))
    # print("q_jac_qdot_error: ", torch.norm(q_jac_qdot - q_jac_qdot_fd) / torch.norm(q_jac_qdot))
    # print("q_jac_tau_error: ", torch.norm(q_jac_tau - q_jac_tau_fd) / torch.norm(q_jac_tau))
    # print("qdot_jac_q_error: ", torch.norm(qdot_jac_q - qdot_jac_q_fd) / torch.norm(qdot_jac_q))
    # print("qdot_jac_qdot_error: ", torch.norm(qdot_jac_qdot - qdot_jac_qdot_fd) / torch.norm(qdot_jac_qdot))
    # print("qdot_jac_tau_error: ", torch.norm(qdot_jac_tau - qdot_jac_tau_fd) / torch.norm(qdot_jac_tau))
# ipdb.set_trace()
# benchmark computations
start_time = time.time()
for i in range(options.runs):
    q_out, qdot_out = cartpole2l.dynamics(q_in, qdot_in, tau_in, h_in)
    q_jac_q, q_jac_qdot, q_jac_tau, qdot_jac_q, qdot_jac_qdot, qdot_jac_tau = cartpole2l.derivatives(q_in, qdot_in, tau_in, h_in) 
end_time = time.time()
torch.cuda.synchronize()
print(end_time - start_time)

# ipdb.set_trace()

# benchmark computations
start_time = time.time()
for i in range(options.runs):
    q_out, qdot_out = cartpole2l.dynamics(q_in, qdot_in, tau_in, h_in)
    q_jac_q, q_jac_qdot, q_jac_tau, qdot_jac_q, qdot_jac_qdot, qdot_jac_tau = finite_diff_derivatives(q_in, qdot_in, tau_in, h_in) 
end_time = time.time()
torch.cuda.synchronize()
print(end_time - start_time)

ipdb.set_trace()

#print("q_in:", q_in)
#print("qdot_in:", qdot_in)
#print("tau_in:", tau_in)
#print("\n> q_out: ", q_out)
#print("> qdot_out: ", qdot_out)
#print("\n> q_jac_q: ", q_jac_q.mT)
#print("> q_jac_qdot: ", q_jac_qdot.mT)
#print("> q_jac_tau: ", q_jac_tau.mT)
#print("> qdot_jac_q: ", qdot_jac_q.mT)
#print("> qdot_jac_qdot: ", qdot_jac_qdot.mT)
#print("> qdot_jac_tau: ", qdot_jac_tau.mT)
