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
    
    ControlPoints = [[1.0 1.0 1.0 1.0; 0.0 0.025 0.07500000000000001 0.1],
                     [1.0 0.75 0.25 0.0; 0.1 0.1 0.1 0.1],
                     [0.0 0.0 0.0 0.0; 0.1 0.07500000000000001 0.025 0.0],
                     reverse([1.0 0.75 0.25 0.0; 0.0 0.0 0.0 0.0],dims=2)]
    knots = SA[0 0 0 0.5 1 1 1]
    weights = SVector(1.0,1.0,1.0,1.0)
    center= SA[6L,5L]

    bodies = []
    for i in 1:length(ControlPoints)
        push!(bodies,DynamicBody(NurbsCurve(MMatrix{size(ControlPoints[i])...}(ControlPoints[i]).*L.+center,knots,weights),(0,1)))
    end

    # circle for Turek Hron
    cps = SA[1 1 0 -1 -1 -1  0  1 1.
             0 1 1  1  0 -1 -1 -1 0]*L/2 .+ center .-SA[L/2,0]
    weights = [1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots =   [0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]

    # make a nurbs circle
    # circle = DynamicBody(NurbsCurve(MMatrix(cps),knots,weights),(0,1))
    circle = AutoBody((x,t)->√sum(abs2,x.-center.+SA[L/2,0])-L/2)
    push!(bodies,circle)


    # west = DynamicBody(BSplineCurve(SA[0. 0 0 ; 0.1 0.05 0]*L .+ [6L,5L],degree=2),(0,1))
    # north = DynamicBody(BSplineCurve(SA[1 0.5 0.; 0  0 0]*L .+[6L,5.1L],degree=2),(0,1))
    # east = DynamicBody(BSplineCurve(SA[0. 0 0; 0 0.05 0.1]*L.+[7L,5L],degree=2),(0,1))
    # south= DynamicBody(BSplineCurve(SA[0 0.5 1; 0  0 0]*L .+[6L,5L],degree=2),(0,1))

    # # make a combined body, carefull with winding direction of each curve
    # bodies = [west,north,east,south]
    

    # make a combined body, carefull with winding direction of each curve
    body = ParametricBody(bodies)
    Simulation((16L,12L),(U,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end


function ParametricBodies.measure(body::ParaBodies,x,t)
    d₁,n₁,V₁ = measure(body.bodies[1],x,t)
    for i in 2:length(body.bodies)-1
        dᵢ,nᵢ,Vᵢ = measure(body.bodies[i],x,t)
        if dᵢ>d₁ 
            d₁,n₁,V₁ = dᵢ,nᵢ,Vᵢ
        end
    end
    # last body is union
    dᵢ,nᵢ,Vᵢ = measure(body.bodies[end],x,t)
    if dᵢ<d₁
        d₁,n₁,V₁ = dᵢ,nᵢ,Vᵢ
    end
    return d₁,n₁,V₁
end
function ParametricBodies.sdf(b::ParaBodies,x,t)
    d₁ = sdf(b.bodies[1],x,t)
    for i in 2:length(b.bodies)-1
        d₁ = max(d₁,sdf(b.bodies[i],x,t))
    end
    # last body is union
    d₁ = min(d₁,sdf(b.bodies[end],x,t))
    return d₁
end

# this makes sur the spline extend to infinity
ParametricBodies.notC¹(l::ParametricBodies.NurbsLocator,uv) = false

# intialize
sim = make_sim(;L=16)#mem=CuArray);
t₀,duration,tstep = sim_time(sim),0.1,0.1;

forces = zeros((2,3length(quadPoints[1])))

anim = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the eastground
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U
        
        measure!(sim,t); mom_step!(sim.flow,sim.pois) # evolve Flow
        @inside sim.flow.p[I] = WaterLily.μ₀(sdf(sim.body,loc(0,I),0.0),sim.ϵ)*sim.flow.p[I] # fix pressure
        t += sim.flow.Δt[end]

        # getInterfaceForces!(forces,sim.flow,sim.body,quadPoints)
        # @show forces[1,1:10]
    end

    
    # flood plot
    R = inside(sim.flow.σ)
    @inside sim.flow.σ[I] = sdf(sim.body,loc(0,I),0)
    # @inside sim.flow.σ[I] = measure(sim.body,loc(0,I),0)[1]
    contourf(sim.flow.σ[R]',cmap=:imola10,lw=0.0,legend=true)
    contour!(sim.flow.σ[R]',levels=[0.],color=:black,lw=1.0,legend=false)

    for i ∈ 1:length(sim.body.bodies)-1
        plot!(sim.body.bodies[i].surf;shift=(1.5,0))
        l,u = sim.body.bodies[i].locate.lower,sim.body.bodies[i].locate.upper
        plot!([l[1],l[1],u[1],u[1],l[1]].+1.5,
              [l[2],u[2],u[2],l[2],l[2]].+1.5,color=:black,ls=:dash)
    end
    xlims!(4sim.L,9sim.L); ylims!(4sim.L,6sim.L)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
# save gif
gif(anim, "giiiifff.gif", fps=24)