using WaterLily,ParametricBodies,StaticArrays,CUDA,WriteVTK

function make_sim(;L=32,Re=50000,n=6,m=4,Uinf=1.f0,λ=1.5f0,LD=0.2f0,
                  T=Float32,mem=CUDA.CuArray)
    # some parameters
    D = Int64(L/LD)
    ω = T(λ*Uinf/D*2)

    # helper functions  
    Rot(ϕ) = SA[cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)]
    function map(x,t)
        x₃ = Rot(-ω*t)*SA[x[1]-0.25f0*D*n,x[2]-0.5f0*D*m]
        return SA[x₃[1]+0.25f0*L,abs(x₃[2]+D*0.5f0)] # as map returns a 2D vector, we have an extruded geom
    end

    # the section of the blade
    NACA(s) = 0.18f0*5*(0.2969f0s-0.126f0s^2-0.3516f0s^4+0.2843f0s^6-0.1036f0s^8)
    curve(s,t) = L*SA[(1-s)^2,NACA(1-s)]

    # body and sim
    body = ParametricBody(curve,HashedLocator(curve,(0,1);T,mem);map)
    Simulation((n*D,m*D,16),(Uinf,0f0,0f0),L;U=ω*D*0.5f0,body,
               ν=T(ω*D*0.5*L/Re),perdir=(3,),T,mem)
end

# make a writer with some attributes, need to output to CPU array to save file (|> Array)
vtk_vort(a::Simulation) = (@WaterLily.loop a.flow.f[I,:] .= WaterLily.ω(I,a.flow.u) over I in inside(a.flow.p); a.flow.f |> Array)
vtk_body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a)); a.flow.σ |> Array;)
vtk_lamda(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.λ₂(I, a.flow.u); a.flow.σ |> Array;)

custom_attrib = Dict("ω"=>vtk_vort, "λ₂"=>vtk_lamda,"d"=>vtk_body)# this maps what to write to the name in the file
# make the writer
writer = vtkWriter("wind_tubine"; attrib=custom_attrib)

# make the sim
sim = make_sim(;mem=CuArray);
t₀ = sim_time(sim); duration = 60; tstep = 0.1

@time for tᵢ in range(t₀,t₀+duration;step=tstep)

    # step the flow
    sim_step!(sim,tᵢ,remeasure=true)

    # info and save
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    write!(writer,sim);
end
close(writer)