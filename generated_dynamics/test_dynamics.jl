using Libdl
using LinearAlgebra
import FiniteDiff as FD
using BenchmarkTools
using Plots

function forward_dyn(q, qdot, tau, cont_forward_dynamics)
    qddot = zeros(length(q))
    ccall(cont_forward_dynamics, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}), q, qdot, tau, qddot)
    return qddot
end

function dynamics(x, tau, cont_forward_dynamics)
    q = x[1:Int(length(x)/2)]
    qdot = x[Int(length(x)/2) + 1:end]
    ẋ = [
        qdot
        forward_dyn(q, qdot, tau, cont_forward_dynamics)
    ]
    return ẋ
end

function calc_rk4_manual(q, qdot, u, h, cont_forward_dynamics)
    x = [q; qdot]
    ẋ1 = dynamics(x, u, cont_forward_dynamics)
    ẋ2 = dynamics(x + h/2*ẋ1, u, cont_forward_dynamics)
    ẋ3 = dynamics(x + h/2*ẋ2, u, cont_forward_dynamics)
    ẋ4 = dynamics(x + h*ẋ3, u, cont_forward_dynamics)

    ẋ = (ẋ1 + 2*ẋ2 + 2*ẋ3 + ẋ4)/6;

    x_next = x + h*ẋ
    return x_next[1:Int(length(x)/2)], x_next[Int(length(x)/2) + 1:end]
end

function calc_rk4(q, qdot, u, h, forward_dynamics)
    q_next = zeros(length(q))
    qdot_next = zeros(length(qdot))
    h = [h]
    ccall(forward_dynamics, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}, Ref{Cdouble}),
            q, qdot, u, h, q_next, qdot_next)
    return q_next, qdot_next
end

function calc_rk4_derivatives(q, qdot, u, h, forward_derivatives)
    q_jac_qout = zeros(length(q), length(q))
    q_jac_qdotout = zeros(length(q), length(q))
    q_jac_uout = zeros(length(q), length(u))
    qdot_jac_qout = zeros(length(q), length(q))
    qdot_jac_qdotout = zeros(length(q), length(q))
    qdot_jac_uout = zeros(length(q), length(u))
    h = [h]
    ccall(forward_derivatives, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}, Ref{Cdouble},
        Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}),
            q, qdot, u, h, q_jac_qout, q_jac_qdotout, q_jac_uout, qdot_jac_qout, qdot_jac_qdotout, 
            qdot_jac_uout)
    return q_jac_qout, q_jac_qdotout, q_jac_uout, qdot_jac_qout, qdot_jac_qdotout, qdot_jac_uout
end

# Load library
lib = dlopen(joinpath(@__DIR__, "build/libdynamics.so"))
cont_forward_dynamics = dlsym(lib, :cont_forward_dynamics)
forward_dynamics = dlsym(lib, :forward_dynamics)
forward_derivatives = dlsym(lib, :forward_derivatives)
nq = 2
q, qdot, tau = randn(nq), randn(nq), randn(nq);
# q = [1.1, 2, 3.]
# qdot = [1, 2, 3.]
# tau = [2.0, 0, 1.]
h = 0.01  # large step, large error
q_next1, qdot_next1 = calc_rk4_manual(q, qdot, tau, h, cont_forward_dynamics)
q_next2, qdot_next2 = calc_rk4(q, qdot, tau, h, forward_dynamics)

println(norm(q_next1 - q_next2, Inf))
println(norm(qdot_next1 - qdot_next2, Inf))

J1, J2, J3, J4, J5, J6 = calc_rk4_derivatives(q, qdot, tau, h, forward_derivatives);
println(norm(J1 - FD.finite_difference_jacobian(x_ -> calc_rk4(x_, qdot, tau, h, forward_dynamics)[1], q), Inf))
println(norm(J2 - FD.finite_difference_jacobian(x_ -> calc_rk4(q, x_, tau, h, forward_dynamics)[1], qdot), Inf))
println(norm(J3 - FD.finite_difference_jacobian(x_ -> calc_rk4(q, qdot, x_, h, forward_dynamics)[1], tau), Inf))
println(norm(J4 - FD.finite_difference_jacobian(x_ -> calc_rk4(x_, qdot, tau, h, forward_dynamics)[2], q), Inf))
println(norm(J5 - FD.finite_difference_jacobian(x_ -> calc_rk4(q, x_, tau, h, forward_dynamics)[2], qdot), Inf))
println(norm(J6 - FD.finite_difference_jacobian(x_ -> calc_rk4(q, qdot, x_, h, forward_dynamics)[2], tau), Inf))

# @show @btime calc_rk4_manual($q, $qdot, $tau, $h, $cont_forward_dynamics);
# @show @btime calc_rk4($q, $qdot, $tau, $h, $forward_dynamics);
# @show @btime calc_rk4_derivatives($q, $qdot, $tau, $h, $forward_derivatives);

display(q_next2)
display(qdot_next2)
# display(J1)
# display(J2)
# display(J3)
# display(J4)
# display(J5)
# display(J6)

