using WaterLily
using ParametricBodies
using StaticArrays
using Plots
using CUDA
# parameters
function dynamicSpline(;L=2^4,Re=250,U =1,ϵ=0.5,thk=2ϵ+√2,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA[-1   0   1
             0.5 0.25 0
             0    0   0]*L .+ [2L,3L,8]

    # needed if control points are moved
    cps_m = MMatrix(cps)
    # weights = SA[1.,1.,1.]
    # knots =   SA[0,0,0,1,1,1.]

    # make a nurbs curve
    # circle = NurbsCurve(cps_m,knots,weights)
    circle = BSplineCurve(cps_m;degree=2)

    # use BDIM-σ distance function, make a body and a Simulation
    dist(p,n)=√(p'*p)-thk/2
    body = DynamicBody(circle,(0,1);dist,mem)
    Simulation((8L,6L,16),(U,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end
L=1
# define a flat plat at and angle of attack
cps = SA[-1   0   1
        0.5 0.25 0
        0    0   0]*L .+ [2L,3L,8]

# needed if control points are moved
cps_m = MMatrix(cps)
# weights = SA[1.,1.,1.]
# knots =   SA[0,0,0,1,1,1.]

# make a nurbs curve
# circle = NurbsCurve(cps_m,knots,weights)
circle = BSplineCurve(cps_m;degree=2)

# use BDIM-σ distance function, make a body and a Simulation
dist(p,n)=√(p'*p)-thk/2
l = NurbsLocator(circle,(0,1))
# body = DynamicBody(circle,(0,1);dist)

# # make a writer with some attributes
# velocity(a::Simulation) = a.flow.u |> Array;
# pressure(a::Simulation) = a.flow.p |> Array;
# body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body); 
#                        a.flow.σ |> Array;)
# vorticity(a::Simulation) = (@inside a.flow.σ[I] = 
#                             WaterLily.curl(3,I,a.flow.u)*a.L/a.U;
#                             a.flow.σ |> Array;)
# custom_attrib = Dict(
#     "Velocity" => velocity,
#     "Pressure" => pressure,
#     "Body" => body,
#     "Vorticity_Z" => vorticity,
# )# this maps what to write to the name in the file

# # intialize
# sim = dynamicSpline()#mem=CuArray);
# t₀,duration,tstep = sim_time(sim),10,0.1;
# wr = vtkWriter("ThreeD_nurbs"; attrib=custom_attrib)

# # run
# anim = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)

#     # update until time tᵢ in the background
#     t = sum(sim.flow.Δt[1:end-1])
#     while t < tᵢ*sim.L/sim.U
#         # random update
#         new_pnts = (SA[-1     0   1
#                         0.5 0.25+0.5*sin(π/4*t/sim.L) 0
#                         0     0   0]*sim.L .+ [2sim.L,3sim.L,8])
#         ParametricBodies.update!(sim.body,new_pnts,sim.flow.Δt[end])
#         measure!(sim,t)
#         mom_step!(sim.flow,sim.pois) # evolve Flow
#         t += sim.flow.Δt[end]
#     end

#     # flood plot
#     # @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
#     # contourf(clamp.(sim.flow.σ,-10,10)',dpi=300,
#     #          color=palette(:RdBu_11), clims=(-10,10), linewidth=0,
#     #          aspect_ratio=:equal, legend=false, border=:none)
#     # plot!(sim.body.surf; add_cp=true)
#     # write data
#     write!(wr, sim)


#     # print time step
#     println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
# end
# # save gif
# # gif(anim, "DynamicBody_flow.gif", fps=24)
# close(wr)