using WaterLily
using ParametricBodies
using StaticArrays
using Plots

function make_sim(;L=2^4,Re=250,U =1,ϵ=0.5,mem=Array)
    
    # make a nurbs curve
    west = DynamicBody(BSplineCurve(SA[0. 0 0 ; 2 1 0]*L .+ [2L,0],degree=2),(0,1))
    north = DynamicBody(BSplineCurve(SA[0.2 0.1 0; 0  0 0]*L .+[2L,2L],degree=1),(0,1))
    east = DynamicBody(BSplineCurve(SA[0. 0 0; 0 1 2]*L.+[2.2L,0],degree=2),(0,1))

    # make a combined body, carefull with winding direction of each curve
    body = ParametricBody([west,north,east])
    Simulation((6L,4L),(U,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end

# this makes sur the spline extend to infinity
ParametricBodies.notC¹(l::ParametricBodies.NurbsLocator,uv) = false

# intialize
sim = make_sim(;L=16)#mem=CuArray);
t₀,duration,tstep = sim_time(sim),10,0.1;

anim = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the eastground
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U
        for i in 1:length(sim.body.bodies)
            i==1 && (deformation = (SA[0.5sin(π/4*t/sim.L) 0.1sin(π/4*t/sim.L) 0; 2 1 0] .+ SA[2,0])*sim.L)
            i==2 && (deformation = (SA[0.2 0.1 0; 0  0 0] .+SA[2+0.5sin(π/4*t/sim.L),2])*sim.L)
            i==3 && (deformation = (SA[0 0.1sin(π/4*t/sim.L) 0.5sin(π/4*t/sim.L); 0 1 2] .+ SA[2.2,0])*sim.L)
            ParametricBodies.update!(sim.body.bodies[i],deformation,sim.flow.Δt[end])
        end
        
        measure!(sim,t); mom_step!(sim.flow,sim.pois) # evolve Flow
        sim.flow.Δt[end] = 0.2 # fix time step
        t += sim.flow.Δt[end]
    end

    
    # flood plot
    R = inside(sim.flow.σ)
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U
    contourf(clamp.(sim.flow.σ[R],-10,10)',dpi=300,
            color=palette(:RdBu_11), clims=(-10,10), linewidth=0,
            aspect_ratio=:equal, legend=false, border=:none,title="t=$tᵢ")

    for i ∈ 1:length(sim.body.bodies)
        plot!(sim.body.bodies[i].surf;shift=(1.5,1.5))
        l,u = sim.body.bodies[i].locate.lower,sim.body.bodies[i].locate.upper
        plot!([l[1],l[1],u[1],u[1],l[1]].+1.5,
              [l[2],u[2],u[2],l[2],l[2]].+1.5,color=:black,ls=:dash)
    end

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
# save gif
gif(anim, "/tmp/tmp.gif", fps=24)