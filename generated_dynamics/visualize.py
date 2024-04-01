from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ipdb

def calc_rk4(q, qdot, tau, h, forward_dynamics):
    q_next = np.zeros(len(q), dtype=np.float64)
    qdot_next = np.zeros(len(qdot), dtype=np.float64)
    _q_next = (c_double * len(q_next))()
    _qdot_next = (c_double * len(qdot_next))()
    _q = (c_double * len(q))(*q)
    _qdot = (c_double * len(qdot))(*qdot)
    _tau = (c_double * len(tau))(*tau)
    h = [h]
    _h = (c_double * len(h))(*h)
    forward_dynamics(_q, _qdot, _tau, _h, _q_next, _qdot_next)
    for i in range(len(q_next)):
        q_next[i] = _q_next[i]
        qdot_next[i] = _qdot_next[i]
    return q_next, qdot_next

def rollout(qhist, q0, qdot0, tau, h, forward_dynamics):
    q = q0
    qdot = qdot0
    for i in range(len(qhist)):
        qhist[i, :] = np.hstack((q, qdot))
        q, qdot = calc_rk4(q, qdot, tau, h, forward_dynamics)

import time    
def animate_cartpole2(X):
    # get a random starting state between min state and max state
    p = {"r_1": 1.0, "r_2": 1.0}

    x0 = X[:,1];  # first state at k = 1

    # this function is defined below
    [p_c, p_1, p_2] = dpc_endpositions(x0[1], x0[2], x0[3], p)

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    print(p)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    # plt.show()
    # plt.draw()
    plt.pause(0.5)

    # timer_handle = plt.text(-0.3, x_max[0], '0.00 s', fontsize=15);
    cart_handle, = plt.plot(p_c[0], p_c[1], 'ks', markersize=20, linewidth=3);
    pole_one_handle, = plt.plot([p_c[0], p_1[0]], [p_c[1], p_1[1]], color=np.array([38,124,185])/255, linewidth=8);
    pole_two_handle, = plt.plot([p_1[0], p_2[0]], [p_1[1], p_2[1]], color=np.array([38,124,185])/255, linewidth=8);

    joint_zero_handle, = plt.plot(p_c[0], p_c[1], 'ko', markersize=5)
    joint_one_handle, = plt.plot(p_1[0], p_1[1], 'ko', markersize=5)
    joint_two_handle, = plt.plot(p_2[0], p_2[1], 'ko', markersize=5)

    for k in range(0, X.shape[1]):
        tic = time.time()
        x = X[:,k]

        [p_c, p_1, p_2] = dpc_endpositions(x[0], x[1], x[2], p)

        # timer_handle.set_text('{:.2f} s'.format(tdata[k]))

        cart_handle.set_data(x[0], 0)

        pole_one_handle.set_data([p_c[0], p_1[0]], [p_c[1], p_1[1]])
        pole_two_handle.set_data([p_1[0], p_2[0]], [p_1[1], p_2[1]])

        joint_zero_handle.set_data(p_c[0], p_c[1])
        joint_one_handle.set_data(p_1[0], p_1[1])
        joint_two_handle.set_data(p_2[0], p_2[1])

        # ipdb.set_trace()

        time.sleep(np.max([0.1, 0]))
        plt.pause(0.0001)
    plt.close(fig)

def dpc_endpositions(q_0, q_1, q_2, p):
    # Returns the positions of cart, first joint, and second joint
    # to draw the black circles
    p_c = np.array([q_0, 0]);
    p_1 = p_c + p["r_1"] * np.array([np.cos(q_1+np.pi/2), np.sin(q_1+np.pi/2)])
    p_2 = p_c + p["r_1"] * np.array([np.cos(q_1+np.pi/2), np.sin(q_1+np.pi/2)]) + p["r_2"] * np.array([np.cos(q_1+np.pi/2+q_2), np.sin(q_1+np.pi/2+q_2)]);
    return p_c, p_1, p_2

if __name__ == "__main__":
    nq = 3
    nqdot = 3
    ntau = 3

    qhist = np.zeros((200, nq+nqdot), dtype=np.float64)
    q = np.array([0.0, np.pi, 0.0], dtype=np.float64)
    qdot = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    tau = np.array([10.0, 0.0, 0.0], dtype=np.float64)
    h = np.float64(0.05)

    # q = np.array([1.1, 2, 3.], dtype=np.float64)
    # qdot = np.array([1, 2, 3.], dtype=np.float64)
    # tau = np.array([2.0, 0, 1.], dtype=np.float64)
    # h = np.float64(0.1)


    lib = CDLL("build/libdynamics.so")
    print(lib)

    q, qdot = calc_rk4(q, qdot, tau, h, lib.forward_dynamics)
    # print(q, qdot)

    rollout(qhist, q, qdot, tau, h, lib.forward_dynamics)
    # print(qhist)

    animate_cartpole2(qhist.T)

