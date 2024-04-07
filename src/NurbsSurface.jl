using StaticArrays
using ParametricBodies: Bd
using Plots: @recipe, @series

struct NurbsBasis{d,V<:AbstractVector,W<:AbstractVector} <: Function
    knots :: V
    wgts :: W
end
function NurbsBasis(T,count,knots,weights)
    @assert count == length(weights) "Invalid NurbsBasis: each control point should have a corresponding weights."
    @assert count < length(knots) "Invalid NurbsBasis: the number of knots should be greater than the number of control points."
    degree = length(knots) - count - 1
    knots = SA{T}[knots...]; weights = SA{T}[weights...]
    degree,NurbsBasis{degree,typeof(knots),typeof(weights)}(knots,weights)
end

struct NurbsSurface{du,dv,A<:AbstractArray,U<:NurbsBasis,V<:NurbsBasis} <: Function
    pnts  :: A
    u :: U
    v :: V
end
function NurbsSurface(pnts,knots,weights)
    (nᵤ,nᵥ,dim),T = size(pnts),promote_type(eltype(pnts),Float32)
    degreeᵤ,u = NurbsBasis(T,nᵤ,knots[1],weights[1])
    degreeᵥ,v = NurbsBasis(T,nᵥ,knots[2],weights[2])
    NurbsSurface{degreeᵤ,degreeᵥ,typeof(pnts),typeof(u),typeof(v)}(pnts,u,v)
end

"""
    (::NurbsSurface)(uv,t)

Evaluate the NURBS surface
- `uv` : A tuple, representing the position along the surface where we want to compute the value of that NURBS
- `t` time is currently unused but needed for ParametricBodies
"""
function (l::NurbsSurface{dᵤ,dᵥ})(uv::Tuple{T,T},t)::SVector where {T,dᵤ,dᵥ}
    pt = zeros(SVector{3,T}); wsum=T(0.0); u,v=uv
    for k in 1:size(l.pnts, 1), j in 1:size(l.pnts, 2)
        (l.u.knots[k]>u && l.v.knots[j]>v) && break
        if (l.u.knots[k+dᵤ+1]≥u && l.v.knots[j+dᵥ+1]≥v)
            prod = Bd(l.u.knots,u,k,Val(dᵤ))*Bd(l.v.knots,v,j,Val(dᵥ))*l.u.wgts[k]*l.v.wgts[j];
            pt+=prod*l.pnts[k,j,:]; wsum+=prod
        end
    end
    pt/wsum
end

Base.@kwdef struct PlotAttributesContolPoints
    # https://docs.juliaplots.org/latest/generated/attributes_series/
    line_z=nothing
    linealpha=nothing
    linecolor=:gray
    linestyle=:solid
    linewidth=:auto
    marker_z=nothing
    markeralpha=nothing
    markercolor=:gray
    markershape=:circle
    markersize=4
    markerstrokealpha=nothing
    markerstrokecolor=:match
    markerstrokestyle=:solid
    markerstrokewidth=1
end

@recipe function f(S::NurbsSurface; controlpoints=(;))
    attributes = PlotAttributesContolPoints(;controlpoints...)
    uv = [(u,v) for u in 0:0.01:1, v in 0:0.01:1]
    a = S.pnts
    @series begin
        primary := false
        marker_z := attributes.marker_z
        markeralpha := attributes.markeralpha
        markercolor := attributes.markercolor
        markershape := attributes.markershape
        markersize := attributes.markersize
        markerstrokealpha := attributes.markerstrokealpha
        markerstrokecolor := attributes.markerstrokecolor
        markerstrokestyle := attributes.markerstrokestyle
        markerstrokewidth := attributes.markerstrokewidth
        seriestype := :scatter
        a[:,:,1],a[:,:,2],a[:,:,3]
    end
    @series begin
        primary := false
        line_z := attributes.line_z
        linealpha := attributes.linealpha
        linecolor := attributes.linecolor
        linestyle := attributes.linestyle
        linewidth := attributes.linewidth
        seriestype := :path
        a[:,:,1],a[:,:,2],a[:,:,3]
    end
    @series begin
        primary := false
        line_z := attributes.line_z
        linealpha := attributes.linealpha
        linecolor := attributes.linecolor
        linestyle := attributes.linestyle
        linewidth := attributes.linewidth
        seriestype := :path
        a[:,:,1]',a[:,:,2]',a[:,:,3]'
    end
    ps = S.(uv,0.0)
    xs = getindex.(ps,1)
    ys = getindex.(ps,2)
    zs = getindex.(ps,3)
    seriestype := :surface
    seriesalpha := 0.5
    delete!(plotattributes, :controlpoints)
    delete!(plotattributes, :division_number)
    xs, ys, zs
end

# mesh size 
m,n = 3,4
points = zeros(m,n,3)
for i in 1:m, j in 1:n
    points[i,j,:] = [i-1,j-1,(i-1)^2*(j-1)^2]
end
pnts = SArray{Tuple{m,n,3}}(points)
d1,d2 = 2,2 # degree of each basis
knots = [[zeros(d1); collect(range(0, m-d1) / (m-d1)); ones(d1)],
         [zeros(d2); collect(range(0, n-d2) / (n-d2)); ones(d2)]]
weights = [ones(m),
           ones(n)]

using Plots
surf = NurbsSurface(pnts,knots,weights)
plot(surf, cmap=:RdBu); plot!(xlabel="X",ylabel="Y",zlabel="Z")


