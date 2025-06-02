using Adapt,KernelAbstractions
"""
    HashedLocator

    - `refine<:Function` Performs bounded Newton root-finding step
    - `lims::NTuple{2,T}:` limits of the `uv` parameter
    - `hash<:AbstractArray{T,2}:` Hash to supply good IC to `refine`
    - `lower::SVector{2,T}:` bottom corner of the hash in ξ-space
    - `step::T:` ξ-resolution of the hash

Type to preform efficient and fairly stable `locate`ing on parametric curves. Newton's method is fast, 
but can be very unstable for general parametric curves. This is mitigated by supplying a close initial 
`uv` guess by interpolating `hash`, and by bounding the derivative adjustment in the Newton `refine`ment.

----

    HashedLocator(curve,lims;t⁰=0,step=1,buffer=2,T=Float32,mem=Array)

Creates HashedLocator by sampling the curve and finding the bounding box. This box is expanded by the amount `buffer`. 
The hash array is allocated to span the box with resolution `step` and initialized using `update!(::,curve,t⁰,samples)`.

Example:

    t = 0.
    curve(θ,t) = SA[cos(θ+t),sin(θ+t)]
    locator = HashedLocator(curve,(0.,2π),t⁰=t,step=0.25)
    @test isapprox(locator(SA[.3,.4],t),atan(4,3),rtol=1e-4)
    @test isapprox(curve(locator(SA[-1.2,.9],t),t),SA[-4/5,3/5],rtol=1e-4)

    body = ParametricBody(curve,locator)
    @test isapprox(sdf(body,SA[-.3,-.4],t),-0.5,rtol=1e-4) # inside hash
    @test isapprox(sdf(body,SA[-3.,-4.],t), 4.0,rtol=2e-2) # outside hash
"""
struct HashedLocator{T,F<:Function,A<:AbstractArray{T,2}} <: AbstractLocator
    refine::F
    lims::NTuple{2,T}
    hash::A
    lower::SVector{2,T}
    step::T
    closed::Bool
end
Adapt.adapt_structure(to, x::HashedLocator) = HashedLocator(x.refine,x.lims,adapt(to,x.hash),x.lower,x.step,x.closed)

function HashedLocator(curve,lims;t⁰=0,step=1,buffer=2,T=Float32,mem=Array,kwargs...)
    # Apply type and get refinement function
    lims,t⁰,step = T.(lims),T(t⁰),T(step)
    closed = curve(first(lims),t⁰)≈curve(last(lims),t⁰)
    f = refine(curve,lims,closed && tangent(curve,first(lims),t⁰)≈tangent(curve,last(lims),t⁰))

    # Get curve's bounding box
    samples = range(lims...,64)
    lower = upper = curve(first(samples),t⁰)
    @assert eltype(lower)==T "`curve` is not type stable"
    @assert isa(curve(first(samples),t⁰),SVector{2,T}) "`curve` doesn't return a 2D SVector"
    for uv in samples
        x = curve(uv,t⁰)
        lower = min.(lower,x)
        upper = max.(upper,x)
    end

    # Allocate hash and struct, and update hash
    hash = fill(first(lims),Int.((upper-lower) .÷ step .+ (1+2buffer))...) |> mem
    l=adapt(mem,HashedLocator{T,typeof(f),typeof(hash)}(f,lims,hash,lower.-buffer*step,step,closed))
    update!(l,curve,t⁰,samples)
end

@inline mymod(x,low,high) = low+mod(x-low,high-low)
function refine(curve,lims,closed)::Function
    # uv⁺ = argmin_uv (X-curve(uv,t))² -> alignment(X,uv⁺,t))=0
    align(X,uv,t) = (X-curve(uv,t))'*tangent(curve,uv,t)
    dalign(X,uv,t) = ForwardDiff.derivative(uv->align(X,uv,t),uv)
    mx = (lims[2]-lims[1])/25; mn = mx/100
    return function(X,uv::T,t) where T # step to alignment root
        a,da = align(X,uv,t),dalign(X,uv,t)
        step = da > -eps(T) ? -copysign(mx,a) : clamp(a/da,-mx,mx)
        new = closed ? mymod(uv-step,lims...) : clamp(uv-step,lims...)
        ifelse(isnan(step),uv,new),abs(new-uv)<mn
    end
end
notC¹(l::HashedLocator,uv) = any(uv.≈l.lims)
eachside(l::HashedLocator,uv,s=√eps(typeof(uv))) = l.closed ? mymod.(uv .+(-s,s),l.lims...) : clamp.(uv .+(-s,s),l.lims...)

"""
    update!(l::HashedLocator,curve,t,samples=l.lims)

Updates `l.hash` for `curve` at time `t` by searching through `samples` and refining.
"""
update!(l::HashedLocator,curve,t,samples=l.lims)=(_update!(get_backend(l.hash),64)(l,curve,samples,t,ndrange=size(l.hash));l)
@kernel function _update!(l::HashedLocator{T},curve,@Const(samples),@Const(t)) where T
    # Map index to physical space
    I = @index(Global,Cartesian)
    x = l.step*(SVector{2,T}(I.I...) .-1)+l.lower

    # Grid search for uv within bounds
    @inline dis2(uv) = (q=x-curve(uv,t); q'*q)
    uv = l.hash[I]; d = dis2(uv)
    for uvᵢ in samples
        dᵢ = dis2(uvᵢ)
        dᵢ<d && (uv=uvᵢ; d=dᵢ)
    end
    
    # Refine estimate with clamped Newton step
    l.hash[I] = l.refine(x,uv,t)[1]
end

"""
    (l::HashedLocator)(x,t)

Estimate the parameter value `uv⁺ = argmin_uv (X-curve(uv,t))²` in two steps:
1. Interploate an initial guess  `uv=l.hash(x)`. Return `uv⁺~uv` if `x` is outside the hash domain.
2. Apply a bounded Newton step `uv⁺≈l.refine(x,uv,t)` to refine the estimate.
"""
function (l::HashedLocator)(x,t)
    # Map location to hash index and clamp to within domain
    hash_index = (x-l.lower)/l.step .+ 1
    clamped = clamp.(hash_index,1,size(l.hash))

    # Get hashed parameter and return if index is outside domain
    uv = l.hash[round.(Int,clamped)...]
    hash_index != clamped && return uv

    # Otherwise, refine estimate
    for _ in 1:5
        uv,done = l.refine(x,uv,t); done && break
    end; uv
end

"""
    HashedBody(curve,lims;T,mem,map)

Creates a `ParametericBody` with `locate=HashedLocator(curve,lims...)`.
"""
function HashedBody(curve,lims::Tuple;T=Float32,map=dmap,kwargs...)
    # Wrap in type safe functions (GPUs are picky)
    wcurve(u::U,t::T) where {U,T} = SVector{2,promote_type(U,T)}(curve(u,t))
    wmap(x::SVector{n,X},t::T) where {n,X,T} = SVector{n,promote_type(X,T)}(map(x,t))

    locate = HashedLocator(wcurve,T.(lims);kwargs...)
    ParametricBody(wcurve,locate;map=wmap,kwargs...)
end
Adapt.adapt_structure(to, x::ParametricBody{T,L}) where {T,L<:HashedLocator} =
    ParametricBody(x.curve,x.dotS,adapt(to,x.locate),x.map,x.scale,x.half_thk,x.boundary)
update!(body::ParametricBody{T,L},t) where {T,L<:HashedLocator} = update!(body.locate,body.curve,T(t))