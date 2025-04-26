using WaterLily,ParametricBodies,StaticArrays,Plots,CUDA
# import WaterLily: interp
# using LinearAlgebra: norm
# using ForwardDiff


# viscous_force(a) = -a.flow.ν*viscous_force(a.flow.u,a.body.curve,a.body.dotS,open(a.body),lims(a.body))
# function viscous_force(u,curve,dotS,opn,lim;N=64,δ=1)
#     @assert length(curve(first(lim)))==2 "integrate(..) can only be used for 2D curves"
#     # integrate NURBS curve to compute integral
#     uv_, w_ = ParametricBodies._gausslegendre(N,typeof(first(lim)))
#     # map onto the (uv) interval, need a weight scalling
#     scale=(last(lim)-first(lim))/2; uv_=CuArray(scale*(uv_.+1)); w_=CuArray(scale*w_)
#     # forces array
#     forces = CuArray(zeros(eltype(first(lim)),2,N))
#     # integrate
#     _viscous!(get_backend(u),64)(forces,curve,dotS,u,uv_,opn,δ,ndrange=N)
#     # sum up the forces, automatic dot product
#     forces*w_ |> Array
# end
# @kernel function _viscous!(forces,@Const(curve),@Const(dotS),@Const(u),@Const(uv),@Const(open),@Const(δ))
#     # get index
#     I = @index(Global)
#     s = uv[I]
#     # physical point, diff length, unit normal
#     xᵢ = curve(s); dl = norm(ForwardDiff.derivative(m->curve(m),s))
#     nᵢ = ParametricBodies.perp(curve,s,0); nᵢ /= √(nᵢ'*nᵢ); dx = nᵢ*δ
#     # value on the curve
#     # vᵢ = dotS(s,0)
#     # vᵢ = vᵢ - (vᵢ'*nᵢ)*nᵢ # tangential comp
#     # uᵢ = interp(xᵢ+dx,u) # prop in the field
#     # wᵢ = interp(xᵢ-dx,u) # prop in the field
#     # uᵢ = uᵢ - (uᵢ'*nᵢ)*nᵢ # tangential comp
#     # compute value at integration point, open curve add contribution from other side
#     # forces[:,I] .= (uᵢ-vᵢ).*nᵢ.*dl/δ - open*(wᵢ-uᵢ).*nᵢ.*dl/δ
# end
# function pressure_force(p,curve,dotS,open,lim;N=64,δ=1)
#     @assert length(curve(first(lim)))==2 "integrate(..) can only be used for 2D curves"
#     # integrate NURBS curve to compute integral
#     uv, w = ParametricBodies._gausslegendre(N,typeof(first(lim)))
#     # map onto the (uv) interval, need a weight scalling
#     scale=(last(lim)-first(lim))/2; uv=CuArray(scale*(uv.+1)); w=CuArray(scale*w)
#     # forces array
#     forces = CuArray(zeros(eltype(first(lim)),2,N))
#     # integrate
#     # _pressure!(get_backend(p),64)(forces,curve,dotS,p,uv,open,δ,ndrange=N)
#     _integrate!(get_backend(p),64)(forces,f_pressure,curve,dotS,p,uv,open,δ,ndrange=N)
#     # sum up the forces, automatic dot product
#     forces*w |> Array
# end
# @kernel function _pressure!(forces,@Const(curve),@Const(dotS),@Const(p),@Const(uv),@Const(open),@Const(δ))
#     # get index
#     I = @index(Global)
#     s = uv[I]
#     # physical point, diff length, unit normal
#     xᵢ = curve(s); dl = norm(ForwardDiff.derivative(m->curve(m),s))
#     nᵢ = ParametricBodies.perp(curve,s,0); nᵢ /= √(nᵢ'*nᵢ); dx = nᵢ*δ
#     # compute value at integration point, open curve add contribution from other side
#     # forces[:,I] .= (interp(xᵢ+dx,p) - open*interp(xᵢ-dx,p)).*nᵢ.*dl
#     forces[:,I] .= f_pressure(s,xᵢ,dx,p,nᵢ,dotS,open).*nᵢ.*dl
# end



# import ParametricBodies: lims
# open(b::ParametricBody{T,L};t=0) where {T,L<:NurbsLocator} =(!all(b.curve(first(b.curve.knots),t).≈b.curve(last(b.curve.knots),t)))
# open(b::ParametricBody{T,L};t=0) where {T,L<:HashedLocator} = (!all(b.curve(first(b.locate.lims),t).≈b.curve(last(b.locate.lims),t)))

# # integrate
# pressure_force(a) = integrate(f_pressure,a.flow.p,a.body.curve,a.body.dotS,open(a.body),lims(a.body))
# viscous_force(a) = -a.flow.ν*integrate(f_viscous,a.flow.u,a.body.curve,a.body.dotS,open(a.body),lims(a.body))

# general integrate function
# function integrate(funct,field,curve,dotS,open,lim;N=64,δ=1)
#     # integrate NURBS curve to compute integral
#     uv, w = ParametricBodies._gausslegendre(N,typeof(first(lim)))
#     # map onto the (uv) interval, need a weight scalling
#     scale=(last(lim)-first(lim))/2; uv=CuArray(scale*(uv.+1)); w=CuArray(scale*w)
#     # forces array
#     forces = CuArray(zeros(eltype(first(lim)),2,N))
#     # integrate
#     _integrate!(get_backend(field),64)(forces,funct,curve,dotS,field,uv,open,δ,ndrange=N)
#     # sum up the forces, automatic dot product
#     forces*w |> Array
# end
# @kernel function _integrate!(forces,@Const(funct),@Const(curve),@Const(dotS),@Const(field),@Const(uv),@Const(open),@Const(δ))
#     # get index
#     I = @index(Global)
#     s = uv[I]
#     # physical point, diff length, unit normal
#     xᵢ = curve(s); dl = norm(ForwardDiff.derivative(m->curve(m),s))
#     nᵢ = ParametricBodies.perp(curve,s,0); nᵢ /= √(nᵢ'*nᵢ); dx = nᵢ*δ
#     # integrate on the curve
#     forces[:,I] .= funct(s,xᵢ,dx,field,nᵢ,dotS,open).*nᵢ.*dl
# end
# gs(x,n) = x - (x'*n)*n # Gram-Schmidt
# grad(u,x,dx,a) = (3u-4interp(x+dx,a)+interp(x+2dx,a))/2norm(dx)
# @inline f_pressure(s,x,dx,p,n,dotS,open) = open ? interp(x+dx,p)-interp(x-dx,p) : interp(x+dx,p)
# @inline f_viscous(s,x,dx,u,n,dotS,open) = open ? gs(grad(dotS(s,0),x,dx,u),n)+gs(grad(dotS(s,0),x,-dx,u),n) : gs(grad(dotS(s,0),x,dx,u),n)


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

mem = Array

# make a sim
sim = circle(mem=CuArray)
ParametricBodies.pressure_force(sim) # check if the pressure force is working
ParametricBodies.viscous_force(sim) # check if the viscous force is working

# # set -up simulations time and time-step for ploting
# duration,tstep = 10, 0.1

# # storage
# pforce,vforce,pmom = [],[],[]

# # run
# @gif for tᵢ in range(0,duration;step=tstep)

#     # update until time tᵢ in the background
#     sim_step!(sim,tᵢ,remeasure=false)

#     # flood plot
#     @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
#     flood(sim.flow.σ,shift=(-2,-1.5),clims=(-8,8), axis=([], false),
#           cfill=:seismic,legend=false,border=:none,size=(8*sim.L,6*sim.L))
#     plot!(sim.body.curve;shift=(1.5,1.5),add_cp=true)

#     # compute and store force
#     # push!(pforce,WaterLily.pressure_force(sim)[1])
#     push!(pforce,pressure_force(sim)[1])
#     # push!(vforce,ParametricBodies.pressure_force(sim)[1])
#     push!(vforce,viscous_force(sim)[1])
#     # push!(pmom,pressure_moment(SA_F32[2sim.L,3sim.L],sim)) # should be zero-ish

#     # print time step
#     println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
# end
# plot(range(0,duration;step=tstep),2pforce,label="New",xlabel="tU/L",ylabel="force")
# plot!(range(0,duration;step=tstep),2vforce,label="Old")