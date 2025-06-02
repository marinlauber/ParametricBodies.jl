using ParametricBodies
using StaticArrays,Test
using CUDA

@testset "ParametricBodies.jl" begin
    curve(θ,t) = SA[cos(θ+t),sin(θ+t)]
    locate(x::SVector{2},t) = atan(x[2],x[1])-t
    body = ParametricBody(curve,locate)

    @test body.curve(body.locate(SA[4.,3.],1),1) == SA[4/5,3/5]

    s = ParametricBodies.tangent(body.curve,π/2,0)
    @test s ≈ SA[-1,0]
    @test ParametricBodies.perp(s) ≈ SA[0,1]
    @test ParametricBodies.align(SA[0.1, 0.5],s) ≈ SA[0, 1] # points in direction n
    @test ParametricBodies.align(SA[0.1,-0.5],s) ≈ SA[0,-1] # aligns with p to point down
    
    @test sdf(body,SA[-.3,-.4],2.) ≈ -0.5
    d,n,V = measure(body,SA[-.75,1],4.)
    @test d ≈ 0.25
    @test n ≈ SA[-3/5, 4/5]
    @test V ≈ SA[-4/5,-3/5]

    # use mapping to double and move circle
    U=0.1; map(x,t)=(x-SA[U*t,0])/2
    body = ParametricBody(curve,locate;map,x₀=SA_F64[0,0])
    d,n,V = measure(body,SA[4U,-2.1],4.)
    @test d ≈ 0.1
    @test n ≈ SA[0,-1]
    @test V ≈ SA[2+U,0]

    # test space-curve with thk=0
    curve3(θ,t) = SA[cos(θ),sin(θ),0]
    locate3(x::SVector{3},t) = atan(x[2],x[1])
    body3 = ParametricBody(curve3,locate3,boundary=false)
    @test [measure(body3,SA[3.,4.,0.],0.)...]≈[4,[3/5,4/5,0],[0,0,0]]
    @test [measure(body3,SA[-.3,-.4,0.],0.)...]≈[0.5,[3/5,4/5,0],[0,0,0]]
    @test [measure(body3,SA[1.,0.,1.],0.)...]≈[1,[0,0,1],[0,0,0]]
    
    # "fast" is ignored (without error) by custom locator
    @test all(measure(body3,SA[1.,0.,1.],0.,fastd²=1) .≈ (1,[0,0,1],[0,0,0]))
end

@testset "SMatrix 2x3" begin
    using Random,LinearAlgebra
    Random.seed!(123)
    for i in 1:10
        a = @SMatrix randn(2, 3)
        b = @SVector randn(2)
        @test Matrix(a)\b ≈ a\b
    end

    a = SA[1.0 1e-10 0.0; 0.0 1.0 1e-10]
    b = SA[1.,1.]
    @test Matrix(a)\b ≈ a\b

    a = SA[1.0 2.0 3.0; 2.0 4.0 6.0]
    b = SA[7.0, 14.0]
    @test Matrix(a)\b ≈ a\b

    
    x = SA_F32[1 0 0; 0 1 0]\SA_F32[0,1]
    @test @allocated(SA_F32[1 0 0; 0 1 0]\SA_F32[0,1]) <400
end

@testset "HashedLocators.jl" begin
    curve(θ,t) = SA[cos(θ+t),sin(θ+t)]
    locator = HashedLocator(curve,(0.,2π),T=Float32)
    @test typeof(locator(SA_F32[0,1],0.f0))==Float32

    t = 0.
    locator = HashedLocator(curve,(0.,2π),t⁰=t,step=0.25,buffer=1)
    @test locator.lower ≈ SA[-1.25,-1.25] rtol=0.01
    @test locator(SA[.3,.4],t) ≈ atan(4,3) rtol=1e-4
    @test curve(locator(SA[-1.2,.9],t),t) ≈ SA[-4/5,3/5] rtol=1e-4

    body = ParametricBody(curve,locator)
    @test sdf(body,SA[-3.,-4.],t) ≈ 4. rtol=1.5e-2 # outside hash

    t = 0.5; ParametricBodies.update!(body,t)
    d,n,V = measure(body,SA[-.75,1],t)
    @test d ≈ 0.25
    @test n ≈ SA[-3/5, 4/5] rtol=1e-4
    @test V ≈ SA[-4/5,-3/5] rtol=1e-4

    if CUDA.functional()
        locator = HashedLocator(curve,(0.,2π),t⁰=t,step=0.25,buffer=1,mem=CuArray)
        x = [SA_F32[.5,.5],SA_F32[.0,.5]] |> CuArray
        t = CUDA.zeros(2) 
        u = locator.(x,t)
        @test u|>Array ≈ [π/4,π/2]
    end

    # use mapping to double, move and rotate circle
    U=0.1; map(x,t)=SA[cos(t) sin(t); -sin(t) cos(t)]*(x-SA[U*t,0])/2
    body = HashedBody((θ,t) -> SA[cos(θ),sin(θ)],(0.,2π);step=0.25,map,T=Float64)
    d,n,V = measure(body,SA[4U,-2.1],4.)
    @test d ≈ 0.1
    @test n ≈ SA[0,-1]
    @test V ≈ SA[2.1+U,0] # rotation from map ∝ r, not R

    # use map to model full circle with only the positive quadrant arc
    step,buffer = 0.2,1
    body = HashedBody(curve,(0.,π/2);step,buffer,map=(x,t)->abs.(x),T=Float64)
    @test body.locate.lower ≈ step*buffer*[-1,-1]
    @test size(body.locate.hash) == (1÷step+1+2buffer,1÷step+1+2buffer) # radius/step+5
    @test [measure(body,SA[-0.3,-0.4],0.)...] ≈ [-0.5,[-3/5,-4/5],[4/5,-3/5]] rtol=1e-4

    # model arc as space-curve with finite thickness
    body = HashedBody(curve,(0.,π/2);step=0.2,buffer=1,T=Float64,thk=0.2,boundary=false)
    @test [measure(body,SA[0.7,-0.4],0.)...] ≈ [0.4,[-3/5,-4/5],[0,1]] rtol=1e-4
    @test measure(body,SA[0.4,0.3],0.)[2] ≈ [-4/5,-3/5] rtol=1e-4
end

using LinearAlgebra,ForwardDiff
function nurbs_circle(T,R=5;center=SA[0,0])
    cps = SA{T}[R R 0 -R -R -R  0  R R
                0 R R  R  0 -R -R -R 0].+center
    weights = SA[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots = SA[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]
    NurbsCurve(cps,knots,weights)
end
@testset "NurbsCurves.jl" begin
    T = Float32
    # make a square
    square = BSplineCurve(SA{T}[5 5 0 -5 -5 -5  0  5 5
                                0 5 5  5  0 -5 -5 -5 0],degree=1)
    @test square(1f0,0) ≈ square(0f0,0) ≈ [5,0]
    @test square(.5f0,0) ≈ [-5,0]

    # check bspline is type stable. Note that using Float64 for cps will break this!
    @test isa(square(0.,0),SVector)
    @test all([eltype(square(zero(T),0))==T for T in (Float32,Float64)])

    # check winding direction
    cross(a,b) = det([a;;b])
    @test all([cross(square(s,0.),square(s+0.1,0.))>0 for s in range(0,.9,10)])

    # check derivatives
    dcurve(u) = ForwardDiff.derivative(u->square(u,0),u)
    @test dcurve(0f0) ≈ [0,40]
    @test dcurve(0.5f0) ≈ [0,-40]

    # Wrap the shape function inside the parametric body class and check measurements
    body = HashedBody(square, (0,1));
    @test [measure(body,SA[1,2],0)...] ≈ [-3,[0,1],[0,0]]
    @test [measure(body,SA[8,2],0)...] ≈ [3,[1,0],[0,0]]

    # Check that HashedLocator works for closed splines
    @test [body.locate(SA_F32[5,s],0) for s ∈ (-2,-1,-0.1)]≈[0.95,0.975,0.9975]

    # NURBS test
    circle = nurbs_circle(T)
    @test isa(circle(0.,0),SVector)
    @test all([eltype(circle(zero(T),0))==T for T in (Float32,Float64)])

    # Wrap the shape function inside the parametric body class and check measurements
    body = HashedBody(circle, (0,1));
    @test [measure(body,SA[-6,0],0)...] ≈ [1,[-1,0],[0,0]] rtol=1e-6
    @test [measure(body,SA[ 5,5],0)...] ≈ [5√2-5,[ √2/2,√2/2],[0,0]] rtol=1e-6
    @test [measure(body,SA[-5,5],0)...] ≈ [5√2-5,[-√2/2,√2/2],[0,0]] rtol=1e-6

    # test interpolation base on the NURBS-book p.367 Ex9.1
    pnts = SA[0. 3. -1. -4  -4.
              0. 4.  4.  0. -3.]
    nurbs,s = interpNurbs(pnts;p=3),ParametricBodies._u(pnts)
    @test s≈[0.,5/17,9/17,14/17,1.]
    @test nurbs.knots≈[0.,0.,0.,0.,28/51,1.,1.,1.,1.]
    @test all(reduce(hcat,nurbs.(s,0.0)).-pnts.<10eps(eltype(pnts)))
end
@testset "NurbsLocator.jl" begin
    # define a circle
    T = Float32
    circle = nurbs_circle(T)

    # Check NurbsLocator
    locate = NurbsLocator(circle)
    @test locate.C¹end # closed & smooth
    @test locate([5,5],0) ≈ 1/8
    body = ParametricBody(circle,locate)
    @test [measure(body,SA[5,5],0)...]≈[5√2-5,[√2/2,√2/2],[0,0]] rtol=1e-6

    # Check GPU locating
    if CUDA.functional()
        x = [SA{T}[5,5],SA{T}[0,5]] |> CuArray
        t = CUDA.zeros(2) 
        u = locate.(x,t)
        @test u|>Array ≈ [1/8,1/4]
    end

    @test [measure(body,SA[5,5],0,fastd²=2)...]≈[5√2-5,[0,0],[0,0]] rtol=1e-6 # inside BBox but outside d²
    @test [measure(body,SA[6,8],0,fastd²=2)...]≈[√10,[0,0],[0,0]] rtol=1e-6   # outside BBox (bounded d²)
    @test [measure(body,SA[6,0],0,fastd²=2)...]≈[1,[1,0],[0,0]] rtol=1e-6     # outside BBox but inside d²

    # Check DynamicNurbsBody
    body = DynamicNurbsBody(circle)
    @test [measure(body,SA[5,5],0)...]≈[5√2-5,[√2/2,√2/2],[0,0]] rtol=1e-6
    @test typeof.(measure(body,SA{T}[5,5],T(0)))==(T,SVector{2,T},SVector{2,T}) # passing in T is type stable
    @test typeof.(measure(body,SA[5.,5.],0.))==(Float64,SVector{2,Float64},SVector{2,Float64}) # promotion works
    @test typeof.(measure(body,SA[5,5],0))==(T,SVector{2,T},SVector{2,T}) broken=true # but passing in Ints give mixed type output...

    body = update!(body, circle.pnts .+T(0.1), T(0.1))
    @test [measure(body,SA[5,5],0)...]≈[4.9√2-5,[√2/2,√2/2],[1,1]] rtol=1e-6
    @test [measure(body,SA[0,0],0)...]≈[0.1√2-5,[-√2/2,-√2/2],[1,1]] rtol=1e-6

    # define a 3D torus with minor radius=1
    cps3 = SA{T}[5 5 0 -5 -5 -5  0  5 5 
                 0 5 5  5  0 -5 -5 -5 0
                 0 0 0  0  0  0  0  0 0]
    circle3 = NurbsCurve(cps3,circle.knots,circle.wgts)
    body3 = ParametricBody(circle3,boundary=false,thk=2)
    @test [measure(body3,SA[3.,4.,2.],0.)...]≈[1,[0,0,1],[0,0,0]]

    # Check GPU
    if CUDA.functional()
        x = [SA_F32[3,4,2],SA_F32[5,5,0]] |> CuArray
        t = CUDA.zeros(2)
        u = body3.locate.(x,t)
        @test u|>Array ≈ [atan(4,3)/2π,1/8] atol=5e-3
        a,b = measure.(Ref(body3),x,t) |> Array
        @test all(a .≈ (1,[0,0,1],[0,0,0]))
        @test all(b .≈ (5√2-6,[√2/2,√2/2,0],[0,0,0]))
    end

    # Check (non)convex corners
    e = √eps(Float32)
    openM = ParametricBody(BSplineCurve(SA_F32[1 1 0 -1 -1;0 1 1/2 1 0],degree=1))
    @test [measure(openM,SA[1-e,-1],0f0)...]≈[-1,[0,1],[0,0]] # just inside end-point
    @test [measure(openM,SA[1+e,-1],0f0)...]≈[ 1,[0,-1],[0,0]] # just outside end-point
    @test [measure(openM,SA[0,e],0f0)...]≈[-0.5,[0,1],[0,0]] # inside concave corner
    @test [measure(openM,SA[2+e,2],0f0)...]≈[√2,[√2/2,√2/2],[0,0]] # outside convex corner
    hashM = HashedBody(BSplineCurve(SA_F32[1 1 0 -1 -1;0 1 1/2 1 0],degree=1),(0,1),step=0.1) # doesn't know where knots are!
    @test [measure(hashM,SA[1-e,-1],0f0)...]≈[-1,[0,1],[0,0]] # end-point still works
    @test [measure(hashM,SA[2+e,2],0f0)...]≈[√2,[√2/2,√2/2],[0,0]] broken = true # "internal" corners may not!!
    closedM = ParametricBody(BSplineCurve(SA_F32[-1 1 1 0 -1 -1;0 0 1 1/2 1 0],degree=1))
    @test [measure(closedM,SA[-2-e,-1],0f0)...]≈[√2,[-√2/2,-√2/2],[0,0]] # outside end-point corner
    closedM2 = ParametricBody(BSplineCurve(SA_F32[-1 1 1 0 -1 -1;0 0 1 1/2 1 0],degree=2)) # still has C⁰ end
    @test [measure(closedM2,SA[-2-e,-1],0f0)...]≈[√2,[-√2/2,-√2/2],[0,0]] # outside end-point corner
    closedM3 = ParametricBody(BSplineCurve(SA_F32[0 1 1 0 -1 -1 0;0 0 1 1/2 1 0 0],degree=2)) # C¹ end
    @test [measure(closedM3,SA[-e,-1],0f0)...]≈[1,[0,-1],[0,0]] # outside end-point corner
end
@testset "Extruded Bodies" begin
    circle = nurbs_circle(Float32,7)
    
    # Make a cylinder
    map(x::SVector{3},t) = SA[x[2],x[3]]
    cylinder = ParametricBody(circle;map,ndims=3)
    @test [measure(cylinder,SA[2,3,6],0)...] ≈ [√45-7,[0,3,6]./√45,[0,0,0]] atol=1e-4

    # Make a sphere
    map(x::SVector{3},t) = SA[x[1],hypot(x[2],x[3])]
    sphere = ParametricBody(circle;map,ndims=3) # define ndims
    @test [measure(sphere,SA[2,3,6],0)...] ≈ [0,[2,3,6]./7,[0,0,0]] atol=1e-4

    # Check GPU
    if CUDA.functional()
        t = CUDA.zeros(2)
        x = [SA_F32[2,3,6],SA_F32[0,3,0]] |> CuArray
        a,b = measure.(Ref(sphere),x,t) |> Array
        @test [a...] ≈ [0,[2,3,6]./7,[0,0,0]] atol=1e-4
        @test all(b .≈ (-4,[0,1,0],[0,0,0]))
    end
end
@testset "PlanarBodies.jl" begin
    T = Float32
    # make a square plate
    square = BSplineCurve(SA{T}[5 5 0 -5 -5 -5  0  5 5
                                0 5 5  5  0 -5 -5 -5 0],degree=1)
    body = PlanarBody(square,(0,1)) # test with HashedLocator
    @test [measure(body,SA[8,2,0],0)...]≈[2-√3/2,[1,0,0],[0,0,0]] rtol=1e-6
    @test [measure(body,SA[2,2,3],0)...]≈[2-√3/2,[0,0,1],[0,0,0]] rtol=1e-6

    body = PlanarBody(square;map=(x,t)->SA[2x[2],2x[3],2x[1]]) # test with NurbsLocator & map
    @test [measure(body,SA[0,8,2],0)...]≈[9/2-√3/2,[0,1,0],[0,0,0]] rtol=1e-6
    @test [measure(body,SA[3,2,2],0)...]≈[2-√3/2,[1,0,0],[0,0,0]] rtol=1e-6
end
using WaterLily
@testset "WaterLily" begin
    function circle_sim(nurbslocate=true,mem=Array,T=Float32)
        circle = nurbs_circle(T)
        body = nurbslocate ? DynamicNurbsBody(circle) : HashedBody(circle,(0,1);T,mem)
        return Simulation((8,8),(1,0),5;body,mem,T)
    end
    for mem in (CUDA.functional() ? [CuArray,Array] : [Array])
        for nurbs in [true,false]
            sim = circle_sim(nurbs,mem); d = sim.flow.σ |> Array
            for I in CartesianIndices((4:6,3:7))
                @test d[I]≈√sum(abs2,WaterLily.loc(0,I))-5 atol=1e-6
            end
            if nurbs # `sim.body=...` requires WaterLily 1.2.0+
                dc = 1f0
                sim.body = update!(sim.body, sim.body.curve.pnts .+ dc, sim.flow.Δt[end])
                measure_sdf!(sim.flow.σ,sim.body); d = sim.flow.σ |> Array
                I = CartesianIndex(5,5)
                @test d[I]≈√sum(abs2,WaterLily.loc(0,I) .- dc)-5 atol=1e-6
                @test all(WaterLily.pressure_force(sim) .≈ 0)
                @test all(WaterLily.viscous_force(sim) .≈ 0)
            end
        end
    end
end
function hydrostatic!(p::AbstractArray{T,D},body;psolver=:MultiLevelPoisson,mem=Array) where {T,D}
    # generate field
    z = zero(p); μ⁰ = zeros(T,(size(p)...,D)) |> mem
    # fill the μ⁰ field
    @WaterLily.loop μ⁰[Ii] = WaterLily.μ₀(sdf(body,loc(Ii,T),0),one(T)) over Ii ∈ CartesianIndices(μ⁰)
    WaterLily.BC!(μ⁰,zero(SVector{D,T}))
    # create Poisson solver
    pois = eval(psolver)(p,μ⁰,z) # the initial solution points to p
    @inside pois.z[I] = WaterLily.∂(1,I,pois.L) # zero V contribution everywhere
    WaterLily.update!(pois); solver!(pois;tol=100eps(T),itmx=32)
end
@testset "Hydrostatic pressure test" begin
    for mem in (CUDA.functional() ? [CuArray,Array] : [Array]), T in [Float32, Float64]
        N = 10
        circle = nurbs_circle(T,N;center=SA{T}[1.5N,1.5N])
        p = zeros(T,(3N,3N)) |> mem
        f = zeros(T,(3N,3N,2)) |> mem
        # make two body and probe the pressure
        body = ParametricBody(circle;T)
        # hydrostatic pressure
        hydrostatic!(p,body;mem=mem)
        # hyrostatic!(p,auto;mem) # get the true hydrostatic pressure inside p
        @test WaterLily.pressure_force(p,f,body,0)./(N^2*π) ≈ [1,0] atol=0.1#3e-3
    end
end