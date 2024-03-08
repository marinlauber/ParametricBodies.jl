using StaticArrays
using ForwardDiff: derivative
using FastGaussQuadrature: gausslegendre
using LinearAlgebra: norm
using RecipesBase: @recipe, @series

"""
    NurbsCurve(pnts,knos,weights)

Define a non-uniform rational B-spline curve.
- `pnts`: A 2D array representing the control points of the NURBS curve
- `knots`: A 1D array of th knot vector of the NURBS curve
- `wgts`: A 1D array of the wight of the pnts of the NURBS curve 
- `d`: The degree of the NURBS curve
- `n`: the spacial dimension of the NURBS curve, n ∈ {2,3}
"""
struct NurbsCurve{n,d,A<:AbstractArray,V<:AbstractVector,W<:AbstractVector} <: Function
    pnts::A
    knots::V
    wgts::W
    vel::Union{Nothing,A}
end
function NurbsCurve(pnts,knots,weights;vel=nothing)
    (dim,count),T = size(pnts),promote_type(eltype(pnts),Float32)
    @assert count == length(weights) "Invalid NURBS: each control point should have a corresponding weights."
    @assert count < length(knots) "Invalid NURBS: the number of knots should be greater than the number of control points."
    degree = length(knots) - count - 1 # the one in the input is not used
    knots = SA{T}[knots...]; weights = SA{T}[weights...]
    NurbsCurve{dim,degree,typeof(pnts),typeof(knots),typeof(weights)}(pnts,knots,weights,vel)
end
Base.copy(n::NurbsCurve) = NurbsCurve(copy(n.pnts),copy(n.knots),copy(n.wgts))

"""
    BSplineCurve(pnts; degree=3)

Define a uniform B-spline curve.
- `pnts`: A 2D array representing the control points of the B-spline curve
- `degree`: The degree of the B-spline curve
Note: An open, uniform knot vector for a degree `degree` B-spline is constructed by default.
"""
function BSplineCurve(pnts;degree=1)
    (dim,count),T = size(pnts),promote_type(eltype(pnts),Float32)
    @assert degree <= count - 1 "Invalid B-Spline: the degree should be less than the number of control points minus 1."
    knots = SA{T}[[zeros(degree); collect(range(0, count-degree) / (count-degree)); ones(degree)]...]
    weights = SA{T}[ones(count)...]
    NurbsCurve{dim,degree,typeof(pnts),typeof(knots),typeof(weights)}(pnts,knots,weights)
end

"""
    (::NurbsCurve)(s,t)

Evaluate the NURBS curve
- `s` : A float, representing the position along the spline where we want to compute the value of that NURBS
- `t` time is currently unused but needed for ParametricBodies
"""
function (l::NurbsCurve{n,d})(u::T,t)::SVector where {T,d,n}
    pt = zeros(SVector{n,T}); wsum=T(0.0)
    for k in 1:size(l.pnts, 2)
        l.knots[k]>u && break
        l.knots[k+d+1]≥u && (prod = Bd(l.knots,u,k,Val(d))*l.wgts[k];
                             pt +=prod*l.pnts[:,k]; wsum+=prod)
    end
    pt/wsum
end
function velocity(l::NurbsCurve{n,d},u::T,t)::SVector where {T,d,n}
    vel = zeros(SVector{n,T}); wsum=T(0.0)
    isnothing(l.vel) && return vel # zero velocity by default
    for k in 1:size(l.vel, 2)
        l.knots[k]>u && break
        l.knots[k+d+1]≥u && (prod = Bd(l.knots,u,k,Val(d))*l.wgts[k];
                             vel +=prod*l.vel[:,k]; wsum+=prod)
    end
    vel/wsum
end

"""
    Bd(knot, u, k, ::Val{d}) where d

Compute the Cox-De Boor recursion for B-spline basis functions.
- `knot`: A Vector containing the knots of the B-Spline, with the knot value `k ∈ [0,1]`.
- `u` : A Float representing the value of the parameter on the curve at which the basis function is computed, `u ∈ [0,1]`
- `k` : An Integer representing which basis function is computed.
- `d`: An Integer representing the order of the basis function to be computed.
"""
Bd(knots, u::T, k, ::Val{0}) where T = Int(knots[k]≤u<knots[k+1] || u==knots[k+1]==1)
function Bd(knots, u::T, k, ::Val{d}) where {T,d}
    ((u-knots[k])/max(eps(T),knots[k+d]-knots[k])*Bd(knots,u,k,Val(d-1))
    +(knots[k+d+1]-u)/max(eps(T),knots[k+d+1]-knots[k+1])*Bd(knots,u,k+1,Val(d-1)))
end
"""
    PForce(surf::NurbsCurve,p::AbstractArray{T},s,δ=2.0) where T

Compute the normal (Pressure) force on the NurbsCurve curve from a pressure field `p`
at the parametric coordinate `s`. Useful to compute the force at an integration point
along the NurbsCurve
"""
NurbsForce(surf,p,s,δ=2.0) = PForce(surf,p,s,δ)
function PForce(surf::NurbsCurve,p::AbstractArray{T},s,δ=2.0) where T
    xᵢ = surf(s,0.0)
    δnᵢ = δ*ParametricBodies.norm_dir(surf,s,0.0); δnᵢ/=√(δnᵢ'*δnᵢ)
    Δpₓ = interp(xᵢ+δnᵢ,p)-interp(xᵢ-δnᵢ,p)
    return -Δpₓ.*δnᵢ
end
function VForce(surf::NurbsCurve,u::AbstractArray{T},s,δ=2.0) where T
    xᵢ = surf(s,0.0)
    δnᵢ = δ*ParametricBodies.norm_dir(surf,s,0.0); δnᵢ/=√(δnᵢ'*δnᵢ)
    vᵢ = velocity(surf,s,0.0); τ = SVector{length(vᵢ),T}(zero(vᵢ))
    vᵢ = vᵢ .- sum(vᵢ.*δnᵢ)*δnᵢ
    for j ∈ [-1,1]
        uᵢ = interp(xᵢ+j*δnᵢ,u)
        uᵢ = uᵢ .- sum(uᵢ.*δnᵢ)*δnᵢ
        τ = τ + (uᵢ.-vᵢ)./δ
    end
    return τ
end
"""
    PForce(surf::NurbsCurve,p::AbstractArray{T}) where T

Compute the total force acting on a NurbsCurve from a pressure field `p`.
"""
pforce(surf::NurbsCurve,p::AbstractArray{T};N=64) where T = 
                       integrate(s->PForce(surf,p,s),surf;N)
vforce(surf::NurbsCurve,u::AbstractArray{T};N=64) where T = 
                       integrate(s->VForce(surf,u,s),surf;N)
"""
    integrate(f(uv),curve;N=64)

integrate a function f(uv) along the curve::NurbsCurve, default is the length of the curve
"""
integrate(curve::NurbsCurve;N=64) = integrate((ξ)->1.0,curve::NurbsCurve;N=64)
function integrate(f::Function,curve::NurbsCurve;N=64)
    # integrate NURBS curve to compute its length
    uv_, w_ = gausslegendre(N)
    # map onto the (0,1) interval, need a weight scalling
    uv_ = (uv_.+1)/2; w_/=2 
    sum([f(uv)*norm(derivative(uv->curve(uv,0.),uv))*w for (uv,w) in zip(uv_,w_)])
end
"""
    f(C::NurbsCurve, N::Integer=100)

Plot `recipe`` for `NurbsCurve``, plot the `NurbsCurve` and the control points.
"""
@recipe function f(C::NurbsCurve, N::Integer=100; add_cp=true, shift=[0.,0.])
    seriestype := :path
    primary := false
    @series begin
        linecolor := :black
        linewidth := 2
        markershape := :none
        c = [C(s,0.0) for s ∈ 0:1/N:1]
        getindex.(c,1).+shift[1],getindex.(c,2).+shift[2]
    end
    @series begin
        linewidth  --> (add_cp ? 1 : 0)
        markershape --> (add_cp ? :circle : :none)
        markersize --> (add_cp ? 4 : 0)
        delete!(plotattributes, :add_cp)
        C.pnts[1,:].+shift[1],C.pnts[2,:].+shift[2]
    end
end