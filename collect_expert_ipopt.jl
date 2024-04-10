import Pkg
Pkg.activate(joinpath(@__DIR__))

import MathOptInterface as MOI
import Ipopt
import FiniteDiff as FD
# import ForwardDiff
# import Convex as cvx 
# import Rotations as rot
using LinearAlgebra
# using Plots
using Random
using Libdl
using Printf

include(joinpath(@__DIR__, "utils", "fmincon.jl"))
include(joinpath(@__DIR__, "simple_altro.jl"))

##

#---------------------THIS IS WHAT YOU NEED TO INPUT--------
function discrete_dynamics(p::NamedTuple, x, u, k)
    nq = Int(p.nx / 2)
    q_next = zeros(nq)
    qdot_next = zeros(nq)
    h = [p.dt]
    q = x[1:nq]
    qdot = x[nq+1:end]
    u = [u; zeros(nq - 1)]

    ccall(p.forward_dynamics, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}, Ref{Cdouble}),
        q, qdot, u, h, q_next, qdot_next)
    return [q_next; qdot_next]
end
function discrete_jacobians(p::NamedTuple, x, u)
    nq = Int(p.nx / 2)
    h = [p.dt]
    q = x[1:nq]
    qdot = x[nq+1:end]
    u = [u; zeros(nq - 1)]
    q_jac_qout = zeros(nq, nq)
    q_jac_qdotout = zeros(nq, nq)
    q_jac_uout = zeros(nq, nq)
    qdot_jac_qout = zeros(nq, nq)
    qdot_jac_qdotout = zeros(nq, nq)
    qdot_jac_uout = zeros(nq, nq)

    ccall(p.forward_derivatives, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}, Ref{Cdouble},
            Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}),
        q, qdot, u, h, q_jac_qout, q_jac_qdotout, q_jac_uout, qdot_jac_qout, qdot_jac_qdotout,
        qdot_jac_uout)
    # x_jac_x is [q_jac_qout q_jac_qdotout; qdot_jac_qout qdot_jac_qdotout]
    x_jac_x = zeros(p.nx, p.nx)
    x_jac_x[1:nq, 1:nq] = q_jac_qout
    x_jac_x[1:nq, nq+1:end] = q_jac_qdotout
    x_jac_x[nq+1:end, 1:nq] = qdot_jac_qout
    x_jac_x[nq+1:end, nq+1:end] = qdot_jac_qdotout
    # x_jac_u is [q_jac_uout; qdot_jac_uout]
    x_jac_u = zeros(p.nx, p.nu)
    x_jac_u[1:nq, :] = q_jac_uout[:, 1]
    x_jac_u[nq+1:end, :] = qdot_jac_uout[:, 1]
    return x_jac_x, x_jac_u
end
function ineq_con_x(p, x)
    [x - p.x_max; -x + p.x_min]
end
function ineq_con_u(p, u)
    [u - p.u_max; -u + p.u_min]
end
function ineq_con_u_jac(params, u)
    FD.jacobian(_u -> ineq_con_u(params, _u), u)
end
function ineq_con_x_jac(p, x)
    FD.jacobian(_x -> ineq_con_x(p, _x), x)
end
##

function dynamics(p::NamedTuple, x, u)
    nq = Int(p.nx / 2)
    qdd = zeros(nq)
    q = x[1:nq]
    qd = x[nq+1:end]
    tau = [u; zeros(nq - 1)]

    ccall(p.cont_forward_dynamics, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
        q, qd, tau, qdd)
    return [qd; qdd]
end

function hermite_simpson(params::NamedTuple, x1::Vector, x2::Vector, u, dt::Real)::Vector
    ẋ1 = dynamics(params, x1, u)
    ẋ2 = dynamics(params, x2, u)
    xm = (1 / 2) * (x1 + x2) + (dt / 8) * (ẋ1 - ẋ2)
    ẋm = dynamics(params, xm, u)
    x1 + (dt / 6) * (ẋ1 + 4 * ẋm + ẋ2) - x2
end

##

function create_idx(nx, nu, N)
    # This function creates some useful indexing tools for Z 
    # x_i = Z[idx.x[i]]
    # u_i = Z[idx.u[i]]

    # Feel free to use/not use anything here.


    # our Z vector is [x0, u0, x1, u1, …, xN]
    nz = (N - 1) * nu + N * nx # length of Z 
    x = [(i - 1) * (nx + nu) .+ (1:nx) for i = 1:N]
    u = [(i - 1) * (nx + nu) .+ ((nx+1):(nx+nu)) for i = 1:(N-1)]

    # constraint indexing for the (N-1) dynamics constraints when stacked up
    c = [(i - 1) * (nx) .+ (1:nx) for i = 1:(N-1)]
    nc = (N - 1) * nx # (N-1)*nx 

    return (nx=nx, nu=nu, N=N, nz=nz, nc=nc, x=x, u=u, c=c)
end

function cartpole_cost(params::NamedTuple, Z::Vector)::Real
    idx, N, xg = params.idx, params.N, params.xg
    Q, R, Qf = params.Q, params.R, params.Qf

    J = 0
    for i = 1:(N-1)
        xi = Z[idx.x[i]]
        ui = Z[idx.u[i]]

        J += 0.5 * (xi - xg)' * Q * (xi - xg)
        J += 0.5 * ui' * R * ui
    end

    xn = Z[idx.x[N]]
    J += 0.5 * (xn - xg)' * Qf * (xn - xg)

    return J
end

function cartpole_dynamics_constraints(params::NamedTuple, Z::Vector)::Vector
    idx, N, dt = params.idx, params.N, params.dt

    # create c in a ForwardDiff friendly way (check HW0)
    c = zeros(eltype(Z), idx.nc)

    for i = 1:(N-1)
        xi = Z[idx.x[i]]
        ui = Z[idx.u[i]]
        xip1 = Z[idx.x[i+1]]
        c[idx.c[i]] = hermite_simpson(params, xi, xip1, ui, dt)
    end
    return c
end

function cartpole_equality_constraint(params::NamedTuple, Z::Vector)::Vector
    N, idx, xic, xg = params.N, params.idx, params.xic, params.xg

    [
        Z[idx.x[1]] - xic;
        Z[idx.x[N]] - xg;
        cartpole_dynamics_constraints(params, Z)
    ]
end

function solve_cartpole_swingup(verbose=true)
    CARTPOLE_PATH = joinpath(@__DIR__, "cartpole2l/")
    lib = dlopen(joinpath(CARTPOLE_PATH, "build/libdynamics.so"))
    cont_forward_dynamics = dlsym(lib, :cont_forward_dynamics)
    forward_dynamics = dlsym(lib, :forward_dynamics)
    forward_derivatives = dlsym(lib, :forward_derivatives)

    # problem size
    nq = 3
    nx = nq * 2
    nu = 1
    dt = 0.03
    tf = 6.0
    t_vec = 0:dt:tf
    N = length(t_vec)-1
    # print(N)

    # LQR cost
    # Q = collect(Diagonal([500; 40; 30; 100]))
    Q = 100 * diagm(ones(nx))
    R = 1 * diagm(ones(nu))
    Qf = 10 * Q

    # indexing 
    idx = create_idx(nx, nu, N)
 
    # initial and goal states
    # xic = [0, pi, 0.2, 0.0]
    # xg = [0, 0, 0, 0.0]
    xic = [0, π, 0, 0.0, 0, 0]

    # Random.seed!(1234)
    xic[1] = (2*rand() - 1) * 0.5     # cart from 2 to 2m
    xic[2] = rand() * 2pi           # angle from 0 to 2pi
    xic[3] = rand() * 2pi           # angle from 0 to 2pi
    xic[4]= (2*rand() - 1) * 0.5    # cart velocity from -2 to 2m/s
    xic[5] = (2*rand() - 1) * 0.5   # angle velocity from -2pi to 2pi rad/s
    xic[6] = (2*rand() - 1) * 0.5   # angle velocity from -2pi to 2pi rad/s
    display(xic)

    xg = [0, 0, 0, 0, 0, 0.0]

    u_min = -300. * ones(nu)
    u_max = 300. * ones(nu)

    # state is x y v θ
    x_min = -20000 * ones(nx)
    x_max = 20000 * ones(nx)

    ncx = 2 * nx
    ncu = 2 * nu

    # primal bounds 
    x_l = -Inf * ones(idx.nz)
    x_u = Inf * ones(idx.nz)
    for i = 1:(N-1)
        x_l[idx.u[i]] .= u_min
        x_u[idx.u[i]] .= u_max
    end

    # inequality constraint bounds (this is what we do when we have no inequality constraints)
    c_l = zeros(0)
    c_u = zeros(0)
    function inequality_constraint(params, Z)
        return zeros(eltype(Z), 0)
    end

    Xref = [deepcopy(xg) for i = 1:N]
    Uref = [zeros(nu) for i = 1:N-1]

    # load all useful things into params 
    params = (Q=Q, R=R, Qf=Qf, xic=xic, xg=xg, dt=dt, nx=nx, nu=nu, N=N, idx=idx, cont_forward_dynamics=cont_forward_dynamics)
    params_altro = (
        nx=nx,
        nu=nu,
        ncx=ncx,
        ncu=ncu,
        N=N,
        Q=Q,
        R=R,
        Qf=Qf,
        u_min=u_min,
        u_max=u_max,
        x_min=x_min,
        x_max=x_max,
        Xref=Xref,
        Uref=Uref,
        dt=dt,
        forward_dynamics=forward_dynamics,
        forward_derivatives=forward_derivatives
    )


    z0 = 0.001 * randn(idx.nz)
    for i in 1:N-1
        if i < N / 2
            z0[idx.x[i]] = 0.001 * randn(idx.nx) + xic
            z0[idx.u[i]] = 0.01 * randn(idx.nu)
        else
            z0[idx.x[i]] = 0.001 * randn(idx.nx) + xg
            z0[idx.u[i]] = 0.01 * randn(idx.nu)
        end
    end
    z0[idx.x[N]] = 0.001 * randn(idx.nx) + xg

    # choose diff type (try :auto, then use :finite if :auto doesn't work)
    diff_type = :finite
    #     diff_type = :finite

    Z = fmincon(cartpole_cost, cartpole_equality_constraint, inequality_constraint,
        x_l, x_u, c_l, c_u, z0, params, diff_type;
        tol=1e1, c_tol=1e-1, max_iters=200, verbose=verbose)

    # pull the X and U solutions out of Z
    X = [Z[idx.x[i]] for i = 1:N]
    X[1] .= xic
    U = [Z[idx.u[i]] for i = 1:(N-1)]
    Xn = deepcopy(X)
    Un = deepcopy(U)

    # P = [zeros(nx, nx) for i = 1:N]   # cost to go quadratic term
    # p = [zeros(nx) for i = 1:N]      # cost to go linear term
    # d = [zeros(nu) for i = 1:N-1]    # feedforward control
    # K = [zeros(nu, nx) for i = 1:N-1] # feedback gain
    # iLQR(params_altro, X, U, P, p, K, d, Xn, Un; atol=1e-5, max_iters=2000, verbose=true, ρ=1e0, ϕ=10.0)

    # Rollout the dynamics
    # for i = 1:N-1
    #     X[i+1] = X[i] + dt * dynamics(params, X[i], U[i])
    # end
    return Xn, Un, t_vec, params
end

X, U, t_vec, params_dircol = solve_cartpole_swingup(true)
display(X)
display(U)
print(length(X))

##

using PyCall
using JLD2


traj = (X, U)
save_object("traj.jld2", traj)

np = pyimport("numpy")
X_np = np.asarray(X)
U_np = np.asarray(U)
np.savez("traj", X_np=X_np, U_np=U_np)