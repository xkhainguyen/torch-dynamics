using Libdl
lib = dlopen(joinpath(pwd(), "libdynamics.so"))
forward_dynamics = dlsym(lib, :forward_dynamics)
q, qdot, tau = randn(3), randn(3), randn(3);
qddot = zeros(3);
ccall(forward_dynamics, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), q, qdot, tau, qddot)
println(qddot)