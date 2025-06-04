using RecipesBase: @recipe, @series

"""
    f(C::NurbsCurve, N::Integer=100)

Plot `recipes` for `NurbsCurve`, plot the `NurbsCurve` and the control points.
"""
@recipe function f(C::NurbsCurve{D}, N::Integer=100; add_cp=true, shift=[0.,0.,0.]) where D
    seriestype := :path
    primary := false
    @series begin
        linecolor := :black
        linewidth := 2
        markershape := :none
        c = [C(s,0.0) for s ∈ 0:1/N:1]
        ntuple(i->getindex.(c,i).+shift[i],D)
    end
    @series begin
        linewidth  --> (add_cp ? 1 : 0)
        markershape --> (add_cp ? :circle : :none)
        markersize --> (add_cp ? 4 : 0)
        delete!(plotattributes, :add_cp)
        ntuple(i->C.pnts[i,:].+shift[i],D)
    end
end

"""
    f(C::ParametricBodies, N::Integer=100)

Plot `recipes` for `ParametricBody`.
"""
@recipe function f(b::AbstractParametricBody, time=0, N::Integer=100; shift=[0.,0.])
    seriestype := :path
    primary := false
    @series begin
        linecolor := :black
        linewidth := 2
        markershape := :none
        c = [-b.map(-b.curve(s,time),time) for s ∈ range(lims(b)...;length=N)]
        getindex.(c,1).+shift[1],getindex.(c,2).+shift[2]
    end
end
