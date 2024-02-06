include("src/NurbsCurves.jl")

pnts = SA[0 0.5 1
          0 0.5 1]
curve = BSplineCurve(pnts;degree=2)

struct NurbsSurface{A<:NurbsCurve} <: Function
    curves::A
    curves2::A
end
×(a::NurbsCurve,b::NurbsCurve) = NurbsSurface(a,b)

function (l::NurbsSurface)(uv::NTuple{T},t)::SVector where {T}
    # pt = zeros(SVector{n,T}); wsum=T(0.0)
    # d1,d2,u,v = 
    # for k in 1:size(l.pnts, 1), m in 1:size(l.pnts, 2)
    #     (l.u.knots[k]>u && l.v.knots[m]>v) && break
    #     l.u.knots[k+d+1]≥u && (prod = Bd(l.u.knots,u,k,Val(d1))*Bd(l.v.knots,v,m,Val(d2))*l.u.wgts[k];
    #                            pt +=prod*l.pnts[:,k]; wsum+=prod)
    # end
    # pt/wsum
    l.curves(uv[1],t) + l.curves2(uv[2],t)
end

surface = curve × curve

@show surface((0.,0.),0.0)
