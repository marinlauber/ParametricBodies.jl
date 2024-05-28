using WaterLily,StaticArrays
using ParametricBodies
using Plots,WriteVTK

function make_sim(;L=32,Re=1e3,St=0.3,αₘ=-π/18,U=1,n=8,m=4,T=Float32,mem=Array)
    # Map from simulation coordinate x to surface coordinate ξ
    nose,pivot = SA[L,0.5f0m*L],SA[0.25f0L,0]
    θ₀ = T(αₘ+atan(π*St)); h₀=T(L); ω=T(π*St*U/h₀)
    function map(x,t)
        back = x[1]>nose[1]+2L       # back body?
        ϕ = back ? 5.5f0 : 0         # phase shift
        S = back ? 3L : 0            # horizontal shift
        θ = θ₀*cos(ω*t+ϕ); R = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]
        h = SA[S,h₀*sin(ω*t+ϕ)]
        ξ = R*(x-nose-h-pivot)+pivot # move to origin and align with x-axis
        return SA[ξ[1],abs(ξ[2])]    # reflect to positive y
    end

    # Define foil using NACA0012 profile equation: https://tinyurl.com/NACA00xx
    NACA(s) = 0.6f0*(0.2969f0s-0.126f0s^2-0.3516f0s^4+0.2843f0s^6-0.1036f0s^8)
    foil(s,t) = L*SA[(1-s)^2,NACA(1-s)]
    body = ParametricBody(foil,(0,1);perdir=(3,),map,T,mem)

    Simulation((n*L,m*L,16),(U,0,0),L;ν=U*L/Re,body,perdir=(3,),T,mem)
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
writer = vtkWriter("tandem"; attrib=custom_attrib)

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
gif(anim, "heaving_tandem_foil.gif", fps=24)