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

    # problem size
    nq = 3
    nx = nq * 2
    nu = 1
    dt = 0.03
    tf = 3.0
    t_vec = 0:dt:tf
    N = length(t_vec)

    # LQR cost
    # Q = collect(Diagonal([500; 40; 30; 100]))
    Q = 10 * diagm(ones(nx))
    R = 1 * diagm(ones(nu))
    Qf = 10 * Q

    # indexing 
    idx = create_idx(nx, nu, N)

    # initial and goal states
    # xic = [0, π, 0, 0.0]
    # xg = [0, 0, 0, 0.0]
    xic = [0, π, 0, 0.0, 0, 0]
    xg = [0, 0, 0, 0, 0, 0.0]

    # load all useful things into params 
    params = (Q=Q, R=R, Qf=Qf, xic=xic, xg=xg, dt=dt, nx=nx, nu=nu, N=N, idx=idx, cont_forward_dynamics=cont_forward_dynamics)

    # primal bounds 
    x_l = -Inf * ones(idx.nz)
    x_u = Inf * ones(idx.nz)
    for i = 1:(N-1)
        x_l[idx.u[i]] .= -1000.0*ones(nu)
        x_u[idx.u[i]] .= 1000.0*ones(nu)
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
        tol=1e-6, c_tol=1e-6, max_iters=10_000, verbose=verbose)

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