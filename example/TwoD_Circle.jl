using WaterLily,ParametricBodies,StaticArrays,Plots,CUDA

# @inline function WaterLily.nds(body::ParametricBody,x,t)
#     #measure(body,loc(i,I,T),t,fastd²=d²)
#     d,n,_ = measure(body,x,t,fastd²=1)
#     n*WaterLily.kern(clamp(d,-1,1))
# end

function circle(;L=2^6,Re=250,U=1,mem=Array,T=Float32)

    # NURBS points, weights and knot vector for a circle
    cps = SA{T}[1 1 0 -1 -1 -1  0  1 1
                0 1 1  1  0 -1 -1 -1 0]*L/2 .+ SA{T}[2L,3L]
    weights = SA{T}[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots =   SA{T}[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]

    # make a nurbs curve and a body for the simulation
    Body = ParametricBody(NurbsCurve(cps,knots,weights))
    Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body=Body,T=T,mem=mem)
end

# make a sim
sim = circle(mem=CuArray)

# set -up simulations time and time-step for ploting
duration,tstep = 10, 0.1

# storage
pforce,vforce,pforce_p,vforce_p = [],[],[],[]

# run
@gif for tᵢ in range(0,duration;step=tstep)

    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=false)

    # flood plot
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(sim.flow.σ,shift=(-2,-1.5),clims=(-8,8), axis=([], false),
          cfill=:seismic,legend=false,border=:none,size=(8*sim.L,6*sim.L))
    plot!(sim.body.curve;shift=(1.5,1.5),add_cp=true)

    # compute and store force
    push!(pforce,WaterLily.pressure_force(sim)[1]/sim.L)
    push!(pforce_p,ParametricBodies.pressure_force(sim)[1]/sim.L)
    push!(vforce,WaterLily.viscous_force(sim)[1]/sim.L)
    push!(vforce_p,ParametricBodies.viscous_force(sim)[1]/sim.L)
    # push!(pmom,pressure_moment(SA_F32[2sim.L,3sim.L],sim)) # should be zero-ish

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
plot(range(0,duration;step=tstep),[2pforce, 2vforce, 2pforce_p, 2vforce_p],
     label=["WL pressure" "WL viscous" "PB pressure" "PB viscous"],
     xlabel="tU/L",ylabel="force")
