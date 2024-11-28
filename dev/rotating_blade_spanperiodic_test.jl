using WaterLily
using StaticArrays
using ParametricBodies
using CUDA
using Statistics
using LinearAlgebra
using ParametricBodies
using Adapt
using ForwardDiff
using JLD2
include("/home/marin/Workspace/WaterLily/examples/TwoD_plots.jl")


# planar parametric body
struct PlanarParametricBody{T,S<:Function,L<:Union{Function,HashedLocator},M<:Function} <: AbstractParametricBody
    surf::S    #ξ = surf(uv,t)
    locate::L  #uv = locate(ξ,t)
    map::M     #ξ = map(x,t)
end
function PlanarParametricBody(surf,locate;map=(x,t)->x,T=Float32)
    # N = length(surf(zero(T),0.))
    # Check input functions
    x,t = zero(SVector{2,T}),T(0); ξ = map(x,t)
    PlanarParametricBody(surf,locate,map,T(ParametricBodies.get_scale(map,x)))
end
x2X(x) = SA[x[1],x[2]]
X2x(X) = SA[X[1],X[2],zero(eltype(X))]
function ParametricBodies.surf_props(body::PlanarParametricBody,x,t)
    # Map x to ξ and locate nearest uv
    ξ = body.map(x,t)
    ζ = x2X(ξ) # this is needed because surf props is used for the sdf
    uv = body.locate(ζ,t)

    # Get normal direction and vector from surf to ξ
    n = ParametricBodies.norm_dir(body.surf,uv,t)
    p = ζ-body.surf(uv,t)

    # Fix direction for C⁰ points, normalize, and get distance
    ParametricBodies.notC¹(body.locate,uv) && p'*p>0 && (n = p)
    n /=  √(n'*n)
    return (ParametricBodies.dis(p,n),n,uv)
end
function ParametricBodies.measure(body::PlanarParametricBody,x,t)
    # Surf props and velocity in ξ-frame
    d,n,uv = ParametricBodies.surf_props(body,x,t)
    X = x2X(x)
    dξdt = ForwardDiff.derivative(t->body.surf(uv,t)-body.map(X,t),t)

    # Convert to x-frame with dξ/dx⁻¹ (d has already been scaled)
    dξdx = ForwardDiff.jacobian(x->body.map(x,t),X)
    return (d,X2x(dξdx\n),X2x(dξdx\dξdt))
end
Adapt.adapt_structure(to, x::PlanarParametricBody{T,F,L}) where {T,F,L<:HashedLocator} =
      PlanarParametricBody(x.surf,adapt(to,x.locate),x.map,x.scale)

PlanarParametricBody(surf,uv_bounds::Tuple;step=1,t⁰=0.,T=Float64,mem=Array,map=(x,t)->x) = 
      adapt(mem,PlanarParametricBody(surf,HashedLocator(surf,uv_bounds;step,t⁰,T,mem);map,T))
      
ParametricBodies.update!(body::PlanarParametricBody{T,F,L},t) where {T,F,L<:HashedLocator} = 
    ParametricBodies.update!(body.locate,body.surf,t)


function make_sim(;L=32,Re=50000,n=6,m=4,T=Float32,mem=CUDA.CuArray)
    Uinf=1f0
    λ=1.5f0
    LD=0.2f0
    D=Int64(L/LD)
    ω=T(λ*Uinf/D*2)

    Rot(ϕ) = SA[cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)]
    function map(x,t)
        #I wait that the flow passes through the domain (maybe useless)
        x₃ = Rot(-ω*t)*SA[x[1]-0.25f0*D*n,x[2]-0.5f0*D*m]
        return SA[x₃[1]+0.25f0*L,abs(x₃[2]+D*0.5f0)]
    end

    # NACA(s) = 0.18f0*5*(0.2969f0s-0.126f0s^2-0.3516f0s^4+0.2843f0s^6-0.1036f0s^8)
    # curve(s,t) = L*SA[(1-s)^2,NACA(1-s)]
    file = jldopen("/home/marin/Dropbox/Mosquito_hovering_flight/nurbs_female_fit.jld2","r")
    cps = convert.(T,file["nurbs"]["pnts"].*L)
    weights = convert.(T,file["nurbs"]["weights"])
    knots = convert.(T,file["nurbs"]["knots"])
    close(file)
    curve = NurbsCurve(cps,knots,weights)

    # make a ParametricBody from the curve
    body = PlanarParametricBody(curve,(0,1);map,T,mem)

    Simulation((n*D,m*D,16),(Uinf,0f0,0f0),L;U=ω*D*0.5f0,
                ν=Float64(ω*D*0.5*L/Re),perdir=(3,),body,T,mem)
end

# make a simulation
sim = make_sim(;mem=CuArray)
