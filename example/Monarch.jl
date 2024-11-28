using WaterLily,ParametricBodies
using WriteVTK,StaticArrays
# using CUDA
# CUDA.allowscalar(false)

# Define simulation
function monarch(;L=32,U=1,Re=500,T=Float32,mem=Array)
    points = SA{Float32}[0 24 76 132 169 177 176 163 122 88 89  80  64  42  21   4   2  4 0
                         0 -4 -8  -4   9  23  35  46  60 82 88 107 122 130 128 117 103 40 0] |> reverse
    planform = ParametricBodies.interpNurbs(points,p=3)
    function map(x,t)
        θ = π/8-3π/8*sin(U*t/L)
        Rx = SA[1 0 0; 0 cos(θ) sin(θ); 0 -sin(θ) cos(θ)]
        Rx*100*(x/L-SA[0.75,0.1,1.5])
    end
    body=PlanarBody(planform,(0,1);map,mem)
    Simulation((3L,2L,4L),(0.2,0.,-0.2),L;U,ν=U*L/Re,body,T,mem)
end

sim = monarch(L=24,Re=250,mem=Array);  # closer to real-time

# make a writer with some attributes, need to output to CPU array to save file (|> Array)
velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
_body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a)); 
                                     a.flow.σ |> Array;)
lamda(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.λ₂(I, a.flow.u);
                        a.flow.σ |> Array;)
vorticity(a) = (@WaterLily.loop a.flow.f[I,:] .= ω(I,a.flow.u) over I in inside(a.flow.p);
                a.flow.f |> Array)

custom_attrib = Dict(
    "Velocity" => velocity,
    "Pressure" => pressure,
    "Body" => _body,
    "Lambda" => lamda
)# this maps what to write to the name in the file
# make the writer
writer = vtkWriter("Monarch"; attrib=custom_attrib)

foreach(1:100) do frame
    println(frame)
    sim_step!(sim,sim_time(sim)+0.05)
    write!(writer,sim)
end
close(writer)