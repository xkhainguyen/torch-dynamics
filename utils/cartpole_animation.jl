struct Cartpole
    ℓ::Real
    up_is_0::Bool
end

function create_cartpole!(vis, cartpole::Cartpole)
    start = [-.2, -0.75, -.1]*cartpole.ℓ
    len = [0.4, 1.5, 0.2]*cartpole.ℓ
    mc.setobject!(vis[:cart], mc.HyperRectangle(start..., len...))
    mc.setobject!(vis[:pole], mc.Cylinder(mc.Point(0, 0, -cartpole.ℓ), mc.Point(0, 0, cartpole.ℓ), 0.05*cartpole.ℓ))
    mc.setobject!(vis[:a], mc.HyperSphere(mc.Point(0, 0, 0.0), 0.1*cartpole.ℓ))
    mc.setobject!(vis[:b], mc.HyperSphere(mc.Point(0, 0, 0.0), 0.1*cartpole.ℓ))
end

function update_cartpole_transform!(vis, x, cartpole::Cartpole)
    pole_o = 0.3*cartpole.ℓ
    px = x[1]
    θ = x[2]
    mc.settransform!(vis[:cart], mc.Translation([0, px, 0.0]))
    p1 = [pole_o, px, 0]
    if cartpole.up_is_0
        p2 = p1 + cartpole.ℓ*2*[0, sin(θ), cos(θ)] # for vertical = 0 radians
    else
        p2 = p1 + cartpole.ℓ*2*[0, sin(θ), -cos(θ)] # for vertical = π radians
    end
    mc.settransform!(vis[:a], mc.Translation(p1))
    mc.settransform!(vis[:b], mc.Translation(p2))
    if cartpole.up_is_0
        mc.settransform!(vis[:pole], mc.Translation(0.5*(p1 + p2)) ∘ mc.LinearMap(rot.AngleAxis(-θ, 1, 0, 0))) # for vertical = 0 radians
    else
        mc.settransform!(vis[:pole], mc.Translation(0.5*(p1 + p2)) ∘ mc.LinearMap(rot.AngleAxis(θ, 1, 0, 0))) # for vertical = π radians
    end
end

function animate_cartpole(X, dt)
    vis = mc.Visualizer()
    create_cartpole!(vis)
    anim = mc.Animation(floor(Int,1/dt))
    for k = 1:length(X)
        mc.atframe(anim, k) do
            update_cartpole_transform!(vis, X[k])
        end
    end
    mc.setanimation!(vis, anim)
    return mc.render(vis)
end