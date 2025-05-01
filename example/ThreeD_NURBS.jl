using WaterLily,ParametricBodies,StaticArrays,CUDA,WriteVTK

# parameters
function make_sim(;L=2^4,Re=250,U =1,ϵ=1.0,thk=2ϵ+√3,mem=Array,T=Float32)
    # define a flat plat at and angle of attack
    cps = SA{T}[-1   0   1
                0.5 0.25 0
                0    0   0]*L .+ SA{T}[2L,2L,L]

    # needed if control points are moved
    curve = BSplineCurve(cps;degree=2)

    # use BDIM-σ distance function, make a body and a Simulation
    body = DynamicNurbsBody(curve;thk=thk,boundary=false)
    Simulation((8L,4L,2L),(U,0,0),L;U,ν=U*L/Re,body,T,mem,ϵ)
end

# make a writer with some attributes
vtk_u(a::Simulation) = a.flow.u |> Array;
vtk_p(a::Simulation) = a.flow.p |> Array;
vtk_d(a::Simulation) = (measure_sdf!(a.flow.σ, a.body); a.flow.σ |> Array;)
custom_attrib = Dict("u"=>vtk_u, "p"=>vtk_p, "d"=>vtk_d)# this maps what to write to the name in the file

# # intialize
sim = make_sim(;L=64,mem=CuArray);
t₀,duration,tstep = sim_time(sim),10,0.1;
wr = vtkWriter("ThreeD_nurbs_GPU"; attrib=custom_attrib)
Tp = eltype(sim.flow.p)

# run
for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U
        # make it move around
        δx = SA{Tp}[-1   0   1
                    0.5 0.25 0
                    0 0.2sin(2π*t/sim.L) 0.5sin(2π*t/sim.L)]*sim.L .+ SA{Tp}[2sim.L,2sim.L,sim.L]
        sim.body = update!(sim.body,δx,sim.flow.Δt[end])
        # update
        sim_step!(sim;remeasure=true)
        t += sim.flow.Δt[end]
    end

    # print time step
    write!(wr, sim)
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
close(wr)