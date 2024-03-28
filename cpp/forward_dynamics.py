import math
from torch import nn
from torch.autograd import Function
import torch
import argparse

import cartpole2l_cpp

nq = 3
nqdot = 3
ntau = 3

parser = argparse.ArgumentParser()
parser.add_argument('--example', choices=['py', 'cpp', 'cuda'], default='cpp')
parser.add_argument('-b', '--batch-size', type=int, default=1)
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

# q_in = torch.randn(options.batch_size, nq, **kwargs)
# qdot_in = torch.randn(options.batch_size, nqdot, **kwargs)
# tau_in = torch.randn(options.batch_size, ntau, **kwargs)

q_in = torch.tensor([1, 2, 3.], **kwargs)
qdot_in = torch.tensor([1, 2, 3.], **kwargs)
tau_in = torch.tensor([2, 0, 0.], **kwargs)

@torch.compile
def vmap_fd(q_in, qdot_in, tau_in):    
    qddot_out = torch.vmap(cartpole2l_cpp.forward_dynamics)(q_in, qdot_in, tau_in)

if (options.batch_size == 1):
    qddot_out = cartpole2l_cpp.forward_dynamics(q_in, qdot_in, tau_in)
else:
    qddot_out = vmap_fd(q_in, qdot_in, tau_in)

print("q_in:", q_in)
print("qdot_in:", qdot_in)
print("tau_in:", tau_in)
print("> qddot_out: ", qddot_out)
