from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
        # qhist[i, 1] = wrap_2pi(qhist[i, 1])
        # qhist[i, 2] = wrap_2pi(qhist[i, 2])
        q, qdot = calc_rk4(q, qdot, tau, h, forward_dynamics)

def wrap_2pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def anime_cartpole1(pos, force=0):
    # animate cartpole
    def update(frame):
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.plot([pos[frame, 0], pos[frame, 0] + -np.sin(pos[frame, 1])], [0, np.cos(pos[frame, 1])], color='blue', linewidth=4)
        if (force):
            ax.arrow(pos[frame, 0], 0, force[frame], 0, color='green', width=0.05)
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, update, frames=len(pos), interval=30, repeat=True)
    plt.title('Cartpole Animation')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.show()

import time    
def animate_cartpole(X, nq=2):
    # get a random starting state between min state and max state
    params = {"r_1": 1.0, "r_2": 1.0, "nq": nq}


    x0 = X[:nq,1];  # first state at k = 1

    # this function is defined below
    # ipdb.set_trace()
    [p_c, p_1, p_2] = dpc_endpositions(tuple(x0), params)
    

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 2)
    ax.set_title('Cartpole {}-Link Animation'.format(nq-1))
    # plt.show()
    # plt.draw()
    # plt.pause(0.5)

    # timer_handle = plt.text(-0.3, x_max[0], '0.00 s', fontsize=15);
    cart_handle, = plt.plot(p_c[0], p_c[1], 'ks', markersize=20, linewidth=3);
    pole_one_handle, = plt.plot([p_c[0], p_1[0]], [p_c[1], p_1[1]], color=np.array([38,124,185])/255, linewidth=8);
    pole_two_handle, = plt.plot([p_1[0], p_2[0]], [p_1[1], p_2[1]], color=np.array([38,124,185])/255, linewidth=8);

    joint_zero_handle, = plt.plot(p_c[0], p_c[1], 'ko', markersize=5)
    joint_one_handle, = plt.plot(p_1[0], p_1[1], 'ko', markersize=5)
    joint_two_handle, = plt.plot(p_2[0], p_2[1], 'ko', markersize=5)

    for k in range(0, X.shape[1]):
        tic = time.time()
        x = X[:nq,k]

        [p_c, p_1, p_2] = dpc_endpositions(tuple(x), params)

        # timer_handle.set_text('{:.2f} s'.format(tdata[k]))

        cart_handle.set_data(x[0], 0)

        pole_one_handle.set_data([p_c[0], p_1[0]], [p_c[1], p_1[1]])
        pole_two_handle.set_data([p_1[0], p_2[0]], [p_1[1], p_2[1]])

        joint_zero_handle.set_data(p_c[0], p_c[1]);
        joint_one_handle.set_data(p_1[0], p_1[1]);
        joint_two_handle.set_data(p_2[0], p_2[1]);

        time.sleep(np.max([0.1, 0]))
        plt.pause(0.0001)
    plt.close(fig)

def dpc_endpositions(q, p):
    # Returns the positions of cart, first joint, and second joint
    # to draw the black circles
    if p["nq"] == 2:
        q_0, q_1 = q
        q_2 = 0.0
    elif p["nq"] == 3:
        q_0, q_1, q_2 = q
    else:
        raise NotImplementedError
    p_c = np.array([q_0, 0]);
    p_1 = p_c + p["r_1"] * np.array([-np.sin(q_1), np.cos(q_1)])
    p_2 = p_c + p["r_1"] * np.array([-np.sin(q_1), np.cos(q_1)]) + p["r_2"] * np.array([-np.sin(q_1+q_2), np.cos(q_1+q_2)])
    return p_c, p_1, p_2

if __name__ == "__main__":
    nq = 3
    nqdot = nq
    ntau = nq

    # h = np.float64(0.05)
    # qhist = np.zeros((200, nq+nqdot), dtype=np.float64)

    # q = np.array([0.0, np.pi, 0], dtype=np.float64)
    # qdot = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    # tau = np.array([10.0, 0.0, 0.0], dtype=np.float64)

    # # q = np.array([0.0, np.pi], dtype=np.float64)
    # # qdot = np.array([0.0, 0.0], dtype=np.float64)
    # # tau = np.array([-10.0, 0.0], dtype=np.float64)
    

    # # q = np.array([1.1, 2, 3.], dtype=np.float64)
    # # qdot = np.array([1, 2, 3.], dtype=np.float64)
    # # tau = np.array([2.0, 0, 1.], dtype=np.float64)
    # # h = np.float64(0.1)


    # lib = CDLL("build/libdynamics.so")
    # print(lib)

    # # q, qdot = calc_rk4(q, qdot, tau, h, lib.forward_dynamics)
    # # print(q, qdot)

    # rollout(qhist, q, qdot, tau, h, lib.forward_dynamics)
    # print(qhist)

    # anime_cartpole1(qhist[:,:2])
    
    traj = np.load("traj.npz")
    x_ref = traj["X_np"]
    u_ref = np.squeeze(traj["U_np"])

    animate_cartpole(x_ref.T, nq)


