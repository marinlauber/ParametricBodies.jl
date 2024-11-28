using WaterLily
using ParametricBodies
using StaticArrays
using Plots
include("../example/viz.jl")

function make_sim(;L=2^4,Re=250,U =1,ϵ=0.5,mem=Array)
    
    # make a nurbs curve
    west = DynamicNurbsBody(BSplineCurve(SA[0. 0 0 ; 1 0.5 0]*L .+ [2L,0],degree=2))
    north= DynamicNurbsBody(BSplineCurve(SA[0.2 0.1 0; 0  0 0]*L .+[2L,L],degree=1))
    east = DynamicNurbsBody(BSplineCurve(SA[0. 0 0; 0 0.5 1]*L.+[2.2L,0],degree=2))

    # add a circle
    circle = AutoBody((x,t)->√sum(abs2,x.-SA[2.1L,L])-L/2)
    circle2 = AutoBody((x,t)->√sum(abs2,x.-SA[2.1L,1.5L])-L/4)

    # make a combined body, carefull with winding direction of each curve
    body = Bodies([west,north,east,circle],[∩,∩,+])
    # body = Bodies([west,north,east],[∩,∩])
    # body = Bodies([circle,circle2],[+])
    # body = Bodies([circle,circle2],[-])
    Simulation((5L,3L),(U,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end

# this makes sur the spline extend to infinity
# ParametricBodies.notC¹(l::ParametricBodies.NurbsLocator,uv) = false

# intialize
sim = make_sim(;L=32)#mem=CuArray);

let
    # @inside sim.flow.σ[I] = sdf(sim.body,loc(0,I),0)
    @inside sim.flow.σ[I] = measure(sim.body,loc(0,I),0)[1]
    contourf(sim.flow.σ', lw=0., color=:RdBu)
    # for i ∈ 1:length(sim.body.bodies)-1
    #     plot!(sim.body.bodies[i].curve;shift=(1.5,1.5),add_cp=false)
    # end
    contour!(sim.flow.σ',levels=[0.],color=:green,lw=2,ls=:dash)
    plot!()
end
sim_gif!(sim;duration=10,step=0.1,clims=(-10,10))