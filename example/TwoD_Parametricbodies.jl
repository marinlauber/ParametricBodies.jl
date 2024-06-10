using WaterLily
using ParametricBodies
using StaticArrays
using Plots

quadPoints = [[0.05635083268962915, 0.25, 0.44364916731037085, 0.5563508326896291, 0.75, 0.9436491673103709],
              [0.05635083268962915, 0.25, 0.44364916731037085, 0.5563508326896291, 0.75, 0.9436491673103709],
              [0.05635083268962915, 0.25, 0.44364916731037085, 0.5563508326896291, 0.75, 0.9436491673103709]]
using ParametricBodies: _pforce
Index(Qs,i) = sum(length.(Qs[1:i-1]))+1:sum(length.(Qs[1:i]))
function getInterfaceForces!(forces,flow::Flow,body::ParaBodies,quadPoints)
    for (i,b) in enumerate(body.bodies[1:3])
        I = Index(quadPoints,i)
        forces[:,I] .= reduce(hcat,[-1.0*_pforce(b.surf,flow.p,s,WaterLily.time(flow),Val{false}()) for s ∈ quadPoints[i]])
    end
end
mag(I,u) = √sum(ntuple(i->0.25*(u[I,i]+u[I+δ(i,I),i])^2,length(I)))


function make_sim(;L=2^4,Re=250,U =1,ϵ=0.5,mem=Array)
    
    # make a nurbs curve
    west = DynamicBody(BSplineCurve(SA[0. 0 0 ; 2 1 0]*L .+ [6L,0],degree=2),(0,1))
    north = DynamicBody(BSplineCurve(SA[0.2 0.1 0; 0  0 0]*L .+[6L,2L],degree=1),(0,1))
    east = DynamicBody(BSplineCurve(SA[0. 0 0; 0 1 2]*L.+[6.2L,0],degree=2),(0,1))

    # west = DynamicBody(BSplineCurve(SA[0. 0 0 ; 2 1 0]*L .+ [6L,5L],degree=2),(0,1))
    # north = DynamicBody(BSplineCurve(SA[2 1 0.; 0  0 0]*L .+[6L,7L],degree=1),(0,1))
    # east = DynamicBody(BSplineCurve(SA[0. 0 0; 0 1 2]*L.+[8L,5L],degree=2),(0,1))
    # south= DynamicBody(BSplineCurve(SA[0 1. 2; 0  0 0]*L .+[6L,5L],degree=1),(0,1))

    # make a combined body, carefull with winding direction of each curve
    body = ParametricBody([west,north,east])
    Simulation((16L,12L),(U,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end

# this makes sur the spline extend to infinity
ParametricBodies.notC¹(l::ParametricBodies.NurbsLocator,uv) = false

# intialize
sim = make_sim(;L=16)#mem=CuArray);
t₀,duration,tstep = sim_time(sim),5,0.1;

forces = zeros((2,3length(quadPoints[1])))

anim = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the eastground
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U
        for i in 1:length(sim.body.bodies)
            i==1 && (deformation = (SA[0.5sin(π/4*t/sim.L) 0.1sin(π/4*t/sim.L) 0; 2 1 0] .+ SA[6,0])*sim.L)
            i==2 && (deformation = (SA[0.2 0.1 0; 0  0 0] .+SA[6+0.5sin(π/4*t/sim.L),2])*sim.L)
            i==3 && (deformation = (SA[0 0.1sin(π/4*t/sim.L) 0.5sin(π/4*t/sim.L); 0 1 2] .+ SA[6.2,0])*sim.L)
            ParametricBodies.update!(sim.body.bodies[i],deformation,sim.flow.Δt[end])
        end
        # for i in 1:length(sim.body.bodies)          
        #     i==1 && (deformation = (SA[0. 0 0 ; 2 1 0]*sim.L .+ [6sim.L+sim.L*sin(π/8*t/sim.L),5sim.L]))
        #     i==2 && (deformation = (SA[2 1 0.; 0  0 0]*sim.L .+[6sim.L+sim.L*sin(π/8*t/sim.L),7sim.L]))
        #     i==3 && (deformation = (SA[0. 0 0; 0 1 2]*sim.L.+[8sim.L+sim.L*sin(π/8*t/sim.L),5sim.L]))
        #     i==4 && (deformation = (SA[0 1. 2; 0  0 0]*sim.L .+[6sim.L+sim.L*sin(π/8*t/sim.L),5sim.L]))
        #     ParametricBodies.update!(sim.body.bodies[i],deformation,sim.flow.Δt[end])
        # end
        
        measure!(sim,t); mom_step!(sim.flow,sim.pois) # evolve Flow
        @inside sim.flow.p[I] = WaterLily.μ₀(sdf(sim.body,loc(0,I),0.0),sim.ϵ)*sim.flow.p[I] # fix pressure
        # sim.flow.Δt[end] = 0.2 # fix time step
        t += sim.flow.Δt[end]

        getInterfaceForces!(forces,sim.flow,sim.body,quadPoints)
        @show forces[1,1:10]
    end

    
    # flood plot
    # R = inside(sim.flow.σ)
    R = CartesianIndices((4sim.L:8sim.L,2:3sim.L))
    # @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U
    contourf(clamp.(sim.flow.σ[R],-10,10)',dpi=300,
            color=palette(:RdBu_11), clims=(-10,10), linewidth=0,
            aspect_ratio=:equal, legend=false, border=:none,title="t=$tᵢ")
    # contourf(clamp.(sim.flow.p[R],-1,1)',dpi=300,
    # @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
    # @inside sim.flow.σ[I] = sdf(sim.body,loc(0,I),0)
    # @inside sim.flow.σ[I] = 0.5*(sim.flow.V[I,1]*(1-sim.flow.μ₀[I,1])+
    #                              sim.flow.V[I+δ(1,I),1]*(1-sim.flow.μ₀[I+δ(1,I),1]))
    # @inside sim.flow.σ[I] = 0.5*((1-sim.flow.μ₀[I,1])+(1-sim.flow.μ₀[I+δ(1,I),1]))
    contourf(clamp.(sim.flow.p[R]',-2,2),dpi=300,
             color=palette(:imola10), clims=(-2,2), linewidth=0,
             aspect_ratio=:equal, legend=false, border=:none,title="t=$tᵢ")
    # @inside sim.flow.σ[I] = mag(I,sim.flow.V)
    # contourf(sim.flow.σ[R]',dpi=300,
    #          color=palette(:RdBu_11), linewidth=0,
    #          aspect_ratio=:equal, legend=true, border=:none, title="t=$tᵢ")
    
    # @inside sim.flow.σ[I] = sdf(sim.body,loc(0,I),0)
    # @inside sim.flow.σ[I] = measure(sim.body,loc(0,I),0)[1]
    # contourf(sim.flow.σ[R]',cmap=:imola10,lw=0.0,legend=false)
    # contour!(sim.flow.σ[R]',levels=[0.],color=:black,lw=1.0,legend=false)

    for i ∈ 1:length(sim.body.bodies)
        plot!(sim.body.bodies[i].surf;shift=(2.5-4sim.L,0))
        # l,u = sim.body.bodies[i].locate.lower,sim.body.bodies[i].locate.upper
        # plot!([l[1],l[1],u[1],u[1],l[1]].+1.5,
            #   [l[2],u[2],u[2],l[2],l[2]].+1.5,color=:black,ls=:dash)
    end

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
# save gif
gif(anim, "/tmp/tmp.gif", fps=24)