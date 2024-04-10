import Pkg
Pkg.activate(joinpath(@__DIR__))

import MathOptInterface as MOI
import Ipopt
import FiniteDiff
# import ForwardDiff
# import Convex as cvx 
# import Rotations as rot
using LinearAlgebra
# using Plots
using Random
using Libdl

include(joinpath(@__DIR__, "utils", "fmincon.jl"))

##
mc = 10; mp1 = 1; mp2 = 1; l1 = 1; l2 = 1; g = 9.81
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

        Ec = 0.5 * mc * xi[4]^2
        Ep1 = 0.5 * mp1 * (l1 * xi[5])^2 + 0.5 * mp1 * g * l1 * cos(xi[2])
        Ep2 = 0.5 * mp2 * (l2 * (xi[5] + xi[6]))^2 + 0.5 * mp2 * g * (l1 * cos(xi[2]) + l2 * cos(xi[2] + xi[3]))
        Ed = mp1 * g * l1 + mp2 * g * (l1 + l2)
        J += params.Qe * (Ec + Ep1 + Ep2 - Ed)^2

        J += 0.5 * ui' * R * ui 
    end

    xn = Z[idx.x[N]]
    J += 0.5 * (xn - xg)' * Qf * (xn - xg)

    Ec = 0.5 * mc * xn[4]^2
    Ep1 = 0.5 * mp1 * (l1 * xn[5])^2 + 0.5 * mp1 * g * l1 * cos(xn[2])
    Ep2 = 0.5 * mp2 * (l2 * (xn[5] + xn[6]))^2 + 0.5 * mp2 * g * (l1 * cos(xn[2]) + l2 * cos(xn[2] + xn[3]))
    Ed = mp1 * g * l1 + mp2 * g * (l1 + l2)
    J += 10 * params.Qe * (Ec + Ep1 + Ep2 - Ed)^2

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
        # Z[idx.x[N]][1] - 0.0;
        # cos(Z[idx.x[N]][2]) - 1.0;
        # cos(Z[idx.x[N]][3]) - 1.0;
        # Z[idx.x[N]][4] - 0.0;
        # Z[idx.x[N]][5] - 0.0;
        # Z[idx.x[N]][6] - 0.0;
        cartpole_dynamics_constraints(params, Z)
    ]
end

function solve_cartpole_swingup(verbose=true)
    CARTPOLE_PATH = joinpath(@__DIR__, "cartpole2l/")
    lib = dlopen(joinpath(CARTPOLE_PATH, "build/libdynamics.so"))
    cont_forward_dynamics = dlsym(lib, :cont_forward_dynamics)

    # problem size
    nq = 3
    nx = nq * 2
    nu = 1
    dt = 0.03
    tf = 7.0
    t_vec = 0:dt:tf
    N = length(t_vec)

    # LQR cost
    # Q = collect(Diagonal([500; 40; 30; 100]))
    Q = 0 * diagm(ones(nx))
    R = 1.0 * diagm(ones(nu))
    Qe = 10.0
    Qf = 10 * Q
    display(Qe)

    # indexing 
    idx = create_idx(nx, nu, N)
 
    # initial and goal states
    # xic = [0, pi, 0.2, 0.0]
    # xg = [0, 0, 0, 0.0]
    xic = [0, π, 0.5, 0.4, 0.4, -0.4]
    Random.seed!(1234)
    xic[1] = (2*rand() - 1) * 0.5     # cart from 2 to 2m
    xic[2] = rand() * 2pi           # angle from 0 to 2pi
    xic[3] = rand() * 2pi           # angle from 0 to 2pi
    xic[4]= (2*rand() - 1) * 0.5    # cart velocity from -2 to 2m/s
    xic[5] = (2*rand() - 1) * 1.0   # angle velocity from -2pi to 2pi rad/s
    xic[6] = (2*rand() - 1) * 1.0   # angle velocity from -2pi to 2pi rad/s
    display(xic)
    xg = [0, 0, 0, 0, 0, 0.0]

    # load all useful things into params 
    params = (Q=Q, R=R, Qf=Qf, Qe=Qe, xic=xic, xg=xg, dt=dt, nx=nx, nu=nu, N=N, idx=idx, cont_forward_dynamics=cont_forward_dynamics)

    # primal bounds 
    x_l = -Inf * ones(idx.nz)
    x_u = Inf * ones(idx.nz)
    for i = 1:(N-1)
        x_l[idx.u[i]] .= -500.0*ones(nu)
        x_u[idx.u[i]] .= 500.0*ones(nu)
    end

    # inequality constraint bounds (this is what we do when we have no inequality constraints)
    c_l = zeros(0)
    c_u = zeros(0)
    function inequality_constraint(params, Z)
        return zeros(eltype(Z), 0)
    end

    z0 = 0.001 * randn(idx.nz)
    for i in 1:N-1
        if i < N / 2
            z0[idx.x[i]] = 0.001 * randn(idx.nx) + xic
            z0[idx.u[i]] = 0.001 * randn(idx.nu)
        else
            z0[idx.x[i]] = 0.001 * randn(idx.nx) + xg
            z0[idx.u[i]] = 0.001 * randn(idx.nu)
        end
    end
    z0[idx.x[N]] = 0.001 * randn(idx.nx) + xg

    # choose diff type (try :auto, then use :finite if :auto doesn't work)
    diff_type = :finite
    #     diff_type = :finite

    Z = fmincon(cartpole_cost, cartpole_equality_constraint, inequality_constraint,
        x_l, x_u, c_l, c_u, z0, params, diff_type;
        tol=1e1, c_tol=1e-1, max_iters=100, verbose=verbose)

    # pull the X and U solutions out of Z
    X = [Z[idx.x[i]] for i = 1:N]
    U = [Z[idx.u[i]] for i = 1:(N-1)]

    # Rollout the dynamics
    # for i = 1:N-1
    #     X[i+1] = X[i] + dt * dynamics(params, X[i], U[i])
    # end
    return X, U, t_vec, params
end

# For short pole
mass_cart = 0.177 # mass of the cart (kg)
mass_pole = 0.076 # mass of the pole (kg)
ℓ = 0.29845 # distance to the center of mass (meters)

X, U, t_vec, params_dircol = solve_cartpole_swingup(true)
display(X)
display(U)
# Xm = hcat(X...)
# Um = hcat(U...)
# display(plot(t_vec, Xm', label=["p" "θ1" "θ2" "ṗ" "θ1̇" "θ2̇"], xlabel="time (s)", title="State Trajectory"))
# display(plot(t_vec[1:end-1], Um', label="", xlabel="time (s)", ylabel="u", title="Controls"))

##

using PyCall
using JLD2



traj = (X, U)
save_object("traj.jld2", traj)

np = pyimport("numpy")
X_np = np.asarray(X)
U_np = np.asarray(U)
np.savez("traj", X_np=X_np, U_np=U_np)