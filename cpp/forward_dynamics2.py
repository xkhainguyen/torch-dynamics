import math
from torch import nn
from torch.autograd import Function
import torch
import argparse

import cartpole2l_cpp
import ipdb

nq = 3
nqdot = 3
ntau = 3
    
class Cartpole2LFunction(Function):
    @staticmethod
    def forward(q_in, qdot_in, tau_in):
        qddot_out = cartpole2l_cpp.forward_dynamics(q_in, qdot_in, tau_in)
        return qddot_out
    
    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        q_in, qdot_in, tau_in = inputs
        qddot_out = output
        ctx.save_for_backward(q_in, qdot_in, tau_in, qddot_out)
    
    @staticmethod
    def vmap(info, in_dims, q_in, qdot_in, tau_in):
        q_in_bdim, qdot_in_bdim, tau_in_bdim  = in_dims

        q_in = q_in.movedim(q_in_bdim, 0)
        qdot_in = qdot_in.movedim(qdot_in_bdim, 0)
        tau_in = tau_in.movedim(tau_in_bdim, 0)

        qddot_out = Cartpole2LFunction.apply(q_in, qdot_in, tau_in)

        return qddot_out, 0

    @staticmethod
    def vmap(info, in_dims, q_in, qdot_in, tau_in):
        q_in_bdim, qdot_in_bdim, tau_in_bdim  = in_dims

        def maybe_expand_bdim_at_front(x, x_bdim):
            if x_bdim is None:
                return x.expand(info.batch_size, *x.shape)
            return x.movedim(x_bdim, 0)

        q_in = maybe_expand_bdim_at_front(q_in, q_in_bdim)
        qdot_in = maybe_expand_bdim_at_front(qdot_in, qdot_in_bdim)
        tau_in = maybe_expand_bdim_at_front(tau_in, tau_in_bdim)

        # ipdb.set_trace()

        qddot_out = Cartpole2LFunction.apply(q_in, qdot_in, tau_in)

        return qddot_out, 0
        
parser = argparse.ArgumentParser()
parser.add_argument('--example', choices=['py', 'cpp', 'cuda'], default='cpp')
parser.add_argument('-b', '--batch-size', type=int, default=2)
parser.add_argument('-r', '--runs', type=int, default=100)
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

q_in = torch.randn(options.batch_size, nq, **kwargs)
qdot_in = torch.randn(options.batch_size, nqdot, **kwargs)
tau_in = torch.randn(options.batch_size, ntau, **kwargs)

# q_in = torch.tensor([1, 2, 3.], **kwargs)
# qdot_in = torch.tensor([1, 2, 3.], **kwargs)
# tau_in = torch.tensor([2, 0, 0.], **kwargs)

func = Cartpole2LFunction

@torch.compile
def vmap_fd(q_in, qdot_in, tau_in):    
    # qddot_out = torch.vmap(Cartpole2LFunction.apply)(q_in, qdot_in, tau_in)
    qddot_out = torch.vmap(func.apply)(q_in, qdot_in, tau_in)
    return qddot_out


if (options.batch_size == 1):
    qddot_out = cartpole2l_cpp.forward_dynamics(q_in, qdot_in, tau_in)
else:
    qddot_out = vmap_fd(q_in, qdot_in, tau_in)

print("q_in:", q_in)
print("qdot_in:", qdot_in)
print("tau_in:", tau_in)
print("> qddot_out: ", qddot_out)
