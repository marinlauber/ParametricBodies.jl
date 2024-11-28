using WaterLily
using ParametricBodies
using StaticArrays
using Adapt
using ForwardDiff
include("../example/viz.jl")

struct PlanarParametricBody{T,S<:Function,L<:Union{Function,HashedLocator},M<:Function} <: AbstractParametricBody
    surf::S    #ξ = surf(uv,t)
    locate::L  #uv = locate(ξ,t)
    map::M     #ξ = map(x,t)
    scale::T   #|dx/dξ| = scale
    transform :: Function
end
function PlanarParametricBody(surf,locate;map=(x,t)->x,transform=(x,t)->x,T=Float64)
    # Check input functions
    x,t = SVector{2,T}(0,0),T(0); ξ = map(x,t)
    PlanarParametricBody(surf,locate,map,T(ParametricBodies.get_scale(map,x)),transform)
end
function measure(body::PlanarParametricBody,x,t)
    # Surf props and velocity in ξ-frame
    d,n,uv = surf_props(body,x,t)
    x = body.transform(x,t)
    dξdt = ForwardDiff.derivative(t->body.surf(uv,t)-body.map(x,t),t)

    # Convert to x-frame with dξ/dx⁻¹ (d has already been scaled)
    dξdx = ForwardDiff.jacobian(x->body.map(x,t),x)
    return (d,dξdx\n/body.scale,dξdx\dξdt)
end
function surf_props(body::PlanarParametricBody,x,t)
    # Map x to ξ and locate nearest uv
    y = body.transform(x,t)
    ξ = body.map(y,t)
    uv = body.locate(ξ,t)

    # Get normal direction and vector from surf to ξ
    n = ParametricBodies.norm_dir(body.surf,uv,t)
    p = ξ-body.surf(uv,t)

    # Fix direction for C⁰ points, normalize, and get distance
    ParametricBodies.notC¹(body.locate,uv) && p'*p>0 && (n = p)
    n /=  √(n'*n)
    return (body.scale*ParametricBodies.dis(p,n),n,uv)
end
Adapt.adapt_structure(to, x::PlanarParametricBody{T,F,L}) where {T,F,L<:HashedLocator} =
    PlanarParametricBody(x.surf,adapt(to,x.locate),x.map,x.scale,x.transform)

PlanarParametricBody(surf,uv_bounds::Tuple;step=1,t⁰=0.,T=Float64,mem=Array,map=(x,t)->x,transform=(x,t)->x) = 
    adapt(mem,PlanarParametricBody(surf,HashedLocator(surf,uv_bounds;step,t⁰,T,mem);map,T,transform))

update!(body::PlanarParametricBody{T,F,L},t) where {T,F,L<:HashedLocator} = 
    update!(body.locate,body.surf,t)

function invert(expr)
    code = @code_typed(expr).first.code
    @show code
    nothing
end

# using ForwardDiff
# function ParametricBodies.measure(body::ParametricBody{T},x,t) where T
#     # get in the ξ-frame
#     y = SA{T}[x[1],√(x[2]^2+x[3]^2)]
#     θ = atan(x[3],x[2])
#     # Surf props and velocity in ξ-frame
#     d,n,uv = ParametricBodies.surf_props(body,y,t)
#     dξdt = ForwardDiff.derivative(t->body.surf(uv,t)-body.map(y,t),t)

#     # Convert to x-frame with dξ/dx⁻¹ (d has already been scaled)
#     dξdx = ForwardDiff.jacobian(x->body.map(y,t),y)

#     # must correct the normal direction and velocity
#     n,v = dξdx\n/body.scale,dξdx\dξdt
#     v = SA[T](v
#     ParametricBodies.sdf(body::ParametricBody{T},x,t) where T = 
#                 Para[1],sin(θ)*v[2],cos(θ)*v[2])
#     n = SA[T](n[1],sin(θ)*n[2],cos(θ)*n[2]); n /=  √(n'*n)
#     return (d,n,v)
# end
# ParametricBodies.sdf(body::ParametricBody{T},x,t) where T = 
#             ParametricBodies.surf_props(body,SA{T}[x[1],√sum(abs2,x[2],x[3])],t)[1]



function make_sim(;L=2^5,Re=250,U=1,mem=Array,T=Float32)

    # NURBS points, weights and knot vector for a circle
    cps = SA{T}[1 1 0 -1 -1 -1  0  1 1
                0 1 1  1  0 -1 -1 -1 0]*L/2 .+ SA{T}[2L,1.5L]
    weights = SA{T}[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots =   SA{T}[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]

    # make a nurbs curve
    circle = NurbsCurve(cps,knots,weights)

    # make a body and a simulation
    Body = ParametricBody(circle,(0,1);T=T,mem=mem)

    Simulation((6L,3L,3L),(U,0,0),L;U,ν=U*L/Re,body=Body,T=T,mem=mem)
end


T = Float64
mem = Array

 # NURBS points, weights and knot vector for a circle
cps = SA{T}[1 1 0 -1 -1 -1  0  1 1
            0 1 1  1  0 -1 -1 -1 0]
weights = SA{T}[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
knots =   SA{T}[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]

# make a nurbs curve
circle = NurbsCurve(cps,knots,weights)

# make a body and a simulation
transform(x,t) = SA{T}[x[1],√(x[2]^2+x[3]^2)]
body = PlanarParametricBody(circle,(0,1);transform,T=T,mem=mem)

x = SA{T}[2.,0.,0.0]
t = 0.f0

y = body.transform(x,t)
ξ = body.map(y,t)
uv = body.locate(ξ,t)

# Get normal direction and vector from surf to ξ
n = ParametricBodies.norm_dir(body.surf,uv,t)
p = ξ-body.surf(uv,t)

# Fix direction for C⁰ points, normalize, and get distance
ParametricBodies.notC¹(body.locate,uv) && p'*p>0 && (n = p)
n /=  √(n'*n)
(body.scale*ParametricBodies.dis(p,n),n,uv)


(ir,it) = only(Base.code_ircode(transform))
code = @code_typed(transform(x,t)).first

# # make a sim
# using CUDA
# sim = make_sim(;mem=CuArray)
# mom_step!(sim.flow,sim.pois)

# # set -up simulations time and time-step for ploting
# t₀ = round(sim_time(sim))
# duration = 1; tstep = 0.1

# mid = Int(size(sim.flow.p,3)÷2)
# # run
# anim = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)

#     # update until time tᵢ in the background
#     sim_step!(sim,tᵢ,remeasure=false)

#     # flood plot
#     # get_omega!(sim);
#     # plot_vorticity(sim.flow.σ, limit=10)
#     @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U
#     Plots.contourf(clamp.(sim.flow.σ[:,:,mid]|>Array,-10,10)',dpi=300,
#                    color=palette(:RdBu_11), clims=(-10, 10), linewidth=0,
#                    aspect_ratio=:equal, legend=false, border=:none)
#     # pforce = ParametricBodies.∮nds(sim.flow.p,sim.body,tᵢ)
#     # vforce = ParametricBodies.∮τnds(sim.flow.u,sim.body,tᵢ)
#     # @show vforce,pforce

#     # print time step
#     println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
# end
# # save gif
# gif(anim, "/tmp/jl_sfawfg.gif", fps=24)


# # measure_sdf!(sim.flow.σ,sim.body,0.0)
