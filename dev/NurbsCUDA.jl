using StaticArrays


struct NurbsCurve{n,d,A<:AbstractArray,V<:AbstractVector,W<:AbstractVector} <: Function
    pnts::A
    temp::A
    knots::V
    wgts::W
end
function NurbsCurve(pnts,knots,weights;f=Array)
    (dim,count),T = size(pnts),promote_type(eltype(pnts),Float32)
    @assert count == length(weights) "Invalid NURBS: each control point should have a corresponding weights."
    @assert count < length(knots) "Invalid NURBS: the number of knots should be greater than the number of control points."
    degree = length(knots) - count - 1 # the one in the input is not used
    knots, weights = knots |> f, weights |> f
    pnts, temp = pnts |> f, zero(pnts) |> f
    NurbsCurve{dim,degree,typeof(pnts),typeof(knots),typeof(weights)}(pnts,temp,knots,weights)
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