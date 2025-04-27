using FastGaussQuadrature: gausslegendre
using LinearAlgebra: norm, cross
import WaterLily: AbstractSimulation,interp
using CUDA

# helper functions
perp(curve,u,t) = perp(tangent(curve,u,t))
"""
    _gausslegendre(N,T)

Compute the Gauss-Legendre quadrature points and weights for N points
"""
function _gausslegendre(N,T)
    x,w = gausslegendre(N)
    convert.(T,x),convert.(T,w)
end
"""
    integrate(f(uv),curve;N=64)

integrate a function f(uv) along the curve
"""
function integrate(funct,field,curve,dotS,open,lim;N=64,δ=1)
    # findout what memory tupe we are using
    mem = typeof(field).name.wrapper
    # integrate NURBS curve to compute integral
    uv, w = _gausslegendre(N,typeof(first(lim)))
    # map onto the (uv) interval, need a weight scalling
    scale=(last(lim)-first(lim))/2; uv=mem(scale*(uv.+1)); w=mem(scale*w)
    # forces array
    forces = similar(field,(2,N))
    # integrate
    _integrate!(get_backend(field),64)(forces,funct,curve,dotS,field,uv,open,δ,ndrange=N)
    # sum up the forces, automatic dot product
    forces*w |> Array
end

@kernel function _integrate!(forces,@Const(funct),@Const(curve),@Const(dotS),@Const(field),@Const(uv),@Const(open),@Const(δ))
    # get index
    I = @index(Global)
    s = uv[I]
    # physical point, diff length, unit normal
    xᵢ = curve(s); dl = norm(ForwardDiff.derivative(m->curve(m),s))
    nᵢ = perp(curve,s,0); nᵢ /= √(nᵢ'*nᵢ); dx = nᵢ*δ
    # integrate on the curve
    forces[:,I] .= funct(s,xᵢ,dx,field,nᵢ,dotS,open).*nᵢ.*dl
end
gs(x,n) = x - (x'*n)*n # Gram-Schmidt
grad(u,x,dx,a) = (3u-4interp(x+dx,a)+interp(x+2dx,a))/2norm(dx)
@inline f_pressure(s,x,dx,p,n,dotS,open) = open ? interp(x+dx,p)-interp(x-dx,p) : interp(x+dx,p)
@inline f_viscous(s,x,dx,u,n,dotS,open) = open ? gs(grad(dotS(s,0),x,dx,u),n)+gs(grad(dotS(s,0),x,-dx,u),n) : gs(grad(dotS(s,0),x,dx,u),n)

# open and close curve require different thratment
open(b::ParametricBody{T,L};t=0) where {T,L<:NurbsLocator} =(!all(b.curve(first(b.curve.knots),t).≈b.curve(last(b.curve.knots),t)))
open(b::ParametricBody{T,L};t=0) where {T,L<:HashedLocator} = (!all(b.curve(first(b.locate.lims),t).≈b.curve(last(b.locate.lims),t)))
lims(b::ParametricBody{T,L};t=0) where {T,L<:NurbsLocator} = (first(b.curve.knots),last(b.curve.knots))
lims(b::ParametricBody{T,L};t=0) where {T,L<:HashedLocator} = b.locate.lims
"""
    pressure_force(p,df,body::AbstractParametricBody,t=0,T;N)

Surface normal pressure integral along the parametric curve(s)
"""
pressure_force(a::AbstractSimulation) = pressure_force(a.flow.p,a.body)
pressure_force(p,body) = integrate(f_pressure,p,body.curve,body.dotS,open(body),lims(body))
"""
viscous_force(u,ν,df,body::AbstractParametricBody,t=0,T;N)

Surface normal pressure integral along the parametric curve(s)
"""
viscous_force(a::AbstractSimulation) = -a.flow.ν.*viscous_force(a.flow.u,a.body)
viscous_force(u,body) = integrate(f_viscous,u,body.curve,body.dotS,open(body),lims(body))
"""
    pressure_moment(x₀,p,df,body::AbstractParametricBody,t=0,T;N)

Surface normal pressure moment integral along the parametric curve(s)
"""
using LinearAlgebra: cross
pressure_moment(x₀,a::AbstractSimulation) = pressure_moment(x₀,a.flow.p,a.body)
function pressure_moment(x₀,p,body)
    @inline function f_pmoment(s,x,dx,p,n,dotS,open)
        r,fp = x-x₀,f_pressure(s,x,dx,p,n,dotS,open)
        return r[1]*fp-r[2]*fp
    end
    integrate(f_pmoment,p,body.curve,body.dotS,open(body),lims(body))
end