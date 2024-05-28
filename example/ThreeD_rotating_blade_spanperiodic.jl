using WaterLily,StaticArrays
using ParametricBodies
using Plots,WriteVTK

function make_sim(;L=32,Re=50000,n=6,m=4,T=Float32,mem=Array)
    Uinf=1f0
    λ=1.5f0
    LD=0.2f0
    D=Int64(L/LD)
    ω=T(λ*Uinf/D*2)

    Rot(ϕ) = SA[cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)]
    function map(x,t)
        # map the the point of rotation and rotate
        x₃ = Rot(-ω*t)*SA[x[1]-0.25f0*D*n,x[2]-0.5f0*D*m]
        return SA[x₃[1]+0.25f0*L,abs(x₃[2]+D*0.5f0)]
    end

    NACA(s) = 0.18f0*5*(0.2969f0s-0.126f0s^2-0.3516f0s^4+0.2843f0s^6-0.1036f0s^8)
    
    curve(s,t) = L*SA[(1-s)^2,NACA(1-s)]
    body = ParametricBody(curve,(0,1);perdir=(3,),map,T,mem)

    Simulation((n*D,m*D,16),(Uinf,0f0,0f0),L;U=ω*D*0.5f0,
                ν=Float64(ω*D*0.5*L/Re),perdir=(3,),body,T,mem)
end

# make a writer with some attributes, need to output to CPU array to save file (|> Array)
vort(a::Simulation) = (@WaterLily.loop sim.flow.f[I,:] .= WaterLily.ω(I,sim.flow.u) over I in inside(sim.flow.p);
                       a.flow.f |> Array)
_body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a)); 
                                     a.flow.σ |> Array;)
lamda(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.λ₂(I, a.flow.u);
                        a.flow.σ |> Array;)

custom_attrib = Dict(
    "vort" => vort,
    "Lambda" => lamda,
    "Body" => _body
)# this maps what to write to the name in the file
# make the writer
writer = vtkWriter("wind_turbine"; attrib=custom_attrib)

# intialize
sim = make_sim()#mem=CuArray);
t₀,duration,tstep = sim_time(sim),10,0.1;

# run
anim = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)
    
    # step the flow
    sim_step!(sim,tᵢ,remeasure=true)

    # flood plot
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U
    Plots.contourf(clamp.(sim.flow.σ[:,:,8],-10,10)',dpi=300,
                   color=palette(:RdBu_11), clims=(-10,10), linewidth=0,
                   aspect_ratio=:equal, legend=false, border=:none)

    # write the field
    write!(writer,sim)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
close(writer)
# save gif
gif(anim, "wind_turbine.gif", fps=24)