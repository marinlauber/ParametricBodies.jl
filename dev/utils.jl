using ParametricBodies
using Adapt
using ForwardDiff
using CUDA

# planar parametric body
struct SpanwiseParametricBody{T,S<:Function,L<:Union{Function,HashedLocator},M<:Function} <: AbstractParametricBody
    surf::S    #ξ = surf(uv,t)
    locate::L  #uv = locate(ξ,t)
    map::M     #ξ = map(x,t)
    scale::T   #|dx/dξ| = scale
    toξ::Function
    fromξ::Function
end
function SpanwiseParametricBody(surf,locate;perdir=(3,),map=(x,t)->x,T=Float32)
    N = length(surf(zero(T),0.))
    # Check input functions
    x,t = zero(SVector{N,T}),T(0); ξ = map(x,t)
    @CUDA.allowscalar uv = locate(ξ,t); p = ξ-surf(uv,t)
    @assert isa(ξ,SVector{N,T}) "map is not type stable"
    @assert isa(uv,T) "locate is not type stable"
    @assert isa(p,SVector{N,T}) "surf is not type stable"

    # x2X(x) = SA[x[1],x[2]]
    # X2x(X) = SA[X[1],X[2],zero(eltype(X))]
    SpanwiseParametricBody(surf,locate,map,T(ParametricBodies.get_scale(map,x)),toξ(x,perdir...),fromξ(x,perdir...))
end
function toξ(x,perdir)
    perdir == 1 && return function(x)
        return SA[x[2],x[3]]
    end
    perdir == 2 && return function(x)
        return SA[x[1],x[3]]
    end
    return function(x)
        return SA[x[1],x[2]]
    end
end
function fromξ(x,perdir)
    perdir == 2 && return function(x)
        SA[zero(eltype(x)),x[1],x[2]]
    end
    perdir == 2 && return function(x)
        SA[x[1],zero(eltype(x)),x[2]]
    end
    return function(x)
        SA[x[1],x[2],zero(eltype(x))]
    end
end
function ParametricBodies.surf_props(body::SpanwiseParametricBody,x,t)
    # Map x to ξ and locate nearest uv
    X = body.toξ(x) # this is needed because surf props is used for the sdf
    ξ = body.map(X,t)
    uv = body.locate(ξ,t)

    # Get normal direction and vector from surf to ξ
    n = ParametricBodies.norm_dir(body.surf,uv,t)
    p = ξ-body.surf(uv,t)

    # Fix direction for C⁰ points, normalize, and get distance
    ParametricBodies.notC¹(body.locate,uv) && p'*p>0 && (n = p)
    n /=  √(n'*n)
    return (body.scale*ParametricBodies.dis(p,n),n,uv)
end
function ParametricBodies.measure(body::SpanwiseParametricBody,x,t)
    # Surf props and velocity in ξ-frame
    d,n,uv = ParametricBodies.surf_props(body,x,t)
    X = body.toξ(x)
    dξdt = ForwardDiff.derivative(t->body.surf(uv,t)-body.map(X,t),t)

    # Convert to x-frame with dξ/dx⁻¹ (d has already been scaled)
    dξdx = ForwardDiff.jacobian(x->body.map(x,t),X)
    return (d,body.fromξ(dξdx\n/body.scale),body.fromξ(dξdx\dξdt))
end
Adapt.adapt_structure(to, x::SpanwiseParametricBody{T,F,L}) where {T,F,L<:HashedLocator} =
      SpanwiseParametricBody(x.surf,adapt(to,x.locate),x.map,x.scale,x.toξ,x.fromξ)

ParametricBody(surf,uv_bounds::Tuple;perdir=(3,),step=1,t⁰=0.,T=Float64,mem=Array,map=(x,t)->x) = 
      adapt(mem,SpanwiseParametricBody(surf,HashedLocator(surf,uv_bounds;step,t⁰,T,mem);perdir,map,T))
      
ParametricBodies.update!(body::SpanwiseParametricBody{T,F,L},t) where {T,F,L<:HashedLocator} = 
    ParametricBodies.update!(body.locate,body.surf,t)
