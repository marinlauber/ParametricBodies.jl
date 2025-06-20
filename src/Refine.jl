"""
    refine(curve,lims,closed)::Function

Returns a `function(X::AbstractVector,u₀,t)` which finds `u⁺ = argmin(d²(u)=|curve(u,t)-X|²)` by solving

    d²′ = (curve(u⁺,t)-X)'*tangent(curve,u⁺,t) = 0

starting from an initial guess `u₀`. The function attempts to Newton step to the root, falling back on 
gradient descent if `d²′′<0`. The resulting minimizer respects `u⁺ ∈ lims` and `closed` curves.

    Note: A good inital guess `u₀` is critical for robustly finding the _global_ minimizer.

    Optional inputs:
    - `itmx=10`: Maximum number of steps to take
    - `stpmx=(lims[2]-lims[1])/20`: Maximum step size
    - `stpmn=stpmx/100`: Minimum step size before the loop will exit
    - `stpmd=stpmx/10`: Step size below which `d² ≥ fastd²` will exit the loop
"""
function refine(curve,lims,closed)::Function
    align(X,u,t) = (curve(u,t)-X)'*tangent(curve,u,t)
    dalign(X,u,t) = ForwardDiff.derivative(u->align(X,u,t),u)
    return function(X,u::T,t;fastd²=Inf,itmx=10,stpmx=(lims[2]-lims[1])/20,stpmn=stpmx/100,stpmd=stpmx/10) where T
        for _ in 1:itmx
            u₀,a,da = u,align(X,u,t),dalign(X,u,t)
            step = da < eps(T) ? -copysign(stpmx,a) : -clamp(a/da,-stpmx,stpmx)
            u = closed ? mymod(u+step,lims...) : clamp(u+step,lims...)
            Δ = min(abs(step),abs(u-u₀))
            (Δ<stpmn || isfinite(fastd²) && Δ<stpmd && sum(abs2,curve(u,t)-X)≥fastd²) && break
        end; u
    end
end
@inline mymod(x,low,high) = low+mod(x-low,high-low)
