using ParametricBodies
using StaticArrays
using ForwardDiff
using BenchmarkTools
using LinearAlgebra
T = Float64

norm(x) = √(x'*x)
@assert norm(SA[1,1]) ≈ √(2)
@assert norm(SA[1,1,1]) ≈ √(3)
@assert norm(SA[1,1,1,1]) ≈ 2

# quaternion rotation matrix, quarterions are organized as q = [iq₂ jq₂ kq₃ q₀]
function R(q::SVector{4,T}) where T
    q/=norm(q) # make sur they are unit carterions
    SA{T}[q[1]^2-q[2]^2-q[3]^2+q[4]^2  2(q[1]*q[2]-q[3]*q[4])       2(q[1]*q[3]+q[2]*q[4])
          2(q[1]*q[2]+q[3]*q[4])      -q[1]^2+q[2]^2-q[3]^2+q[4]^2  2(q[2]*q[3]-q[1]*q[4])
          2(q[1]*q[3]-q[2]*q[4])       2(q[2]*q[3]+q[1]*q[4])      -q[1]^2-q[2]^2+q[3]^2+q[4]^2]
end
@assert all(R(SA[1.,0,0,0]) .≈ [ 1 0 0; 0 -1 0; 0 0 -1]) # rotation around x-axis
@assert all(R(SA[0.,1,0,0]) .≈ [-1 0 0; 0  1 0; 0 0 -1]) # rotation around y-axis
@assert all(R(SA[0.,0,1,0]) .≈ [-1 0 0; 0 -1 0; 0 0  1]) # rotation around z-axis
@assert all(R(SA[0.,0,0,1]) .≈ [ 1 0 0; 0  1 0; 0 0  1]) # identity, no rotation

# quaternion from rotation matrix (rotation matrix are transposed compared to indices)
function rot2quat(R::AbstractArray{T}) where T
    if (R[3,3] < zero(T))
        if (R[1,1] > R[2,2])
            t = one(T) + R[1,1] - R[2,2] - R[3,3]
            q = SA[t, R[2,1]+R[1,2], R[1,3]+R[3,1], R[3,2]-R[2,3]]
        else
            t = one(T) - R[1,1] + R[2,2] - R[3,3]
            q = SA[R[2,1]+R[1,2], t, R[3,2]+R[2,3], R[1,3]-R[3,1]]
        end
    else
        if (R[1,1] < -R[2,2])
            t = one(T) - R[1,1] - R[2,2] + R[3,3]
            q = SA[R[1,3]+R[3,1], R[3,2]+R[2,3], t, R[2,1]-R[1,2]]
        else
            t = one(T) + R[1,1] + R[2,2] + R[3,3]
            q = SA[R[3,2]-R[2,3], R[1,3]-R[3,1], R[2,1]-R[1,2], t]
        end
    end
    q*T(0.5/√t)
end
# check that the rotation matrix is the same as the one obtained from the quaternion
q = collect(1:4); q=q/norm(q); Rs = R(SA[q...])
@assert all(rot2quat(Rs) .≈ q)
q = rand(4); q=q/norm(q); Rs = R(SA[q...])
@assert all(rot2quat(Rs) .≈ q)
@assert all(rot2quat(R(SA[1.,0,0,0])) .≈ [1,0,0,0]) # rotation around x-axis
@assert all(rot2quat(R(SA[0.,1,0,0])) .≈ [0,1,0,0]) # rotation around y-axis
@assert all(rot2quat(R(SA[0.,0,1,0])) .≈ [0,0,1,0]) # rotation around z-axis
@assert all(rot2quat(R(SA[0.,0,0,1])) .≈ [0,0,0,1]) # identity, no rotation

function get_circle()
    # NURBS points, weights and knot vector for a circle
    cps = SA_F32[1 1 0 -1 -1 -1  0  1 1
                 0 1 1  1  0 -1 -1 -1 0]
    weights = SA_F32[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots =   SA_F32[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]
    NurbsCurve(cps,knots,weights)
end
function get_circle½()
    # NURBS points, weights and knot vector for a circle
    cps = SA_F32[1 1 0 -1 -1
                 0 1 1  1  0]
    weights = SA_F32[1.,√2/2,1.,√2/2,1.]
    knots =   SA_F32[0,0,0,1/2,1/2,1,1,1]
    NurbsCurve(cps,knots,weights)
end

# derivative of a rotation matrix
function ∂R(q::NurbsCurve,s,t)
    ForwardDiff.derivative(s->R(q(s,t)),s)
end
# first derivative
function ∂(curve::NurbsCurve,s,t)
    ForwardDiff.derivative(s->curve(s,t),s)
end
# tangent is normalized, should not need a desingularization as `t` is always someting
tangent(curve::NurbsCurve,s,t) = (ts=∂(curve,s,t); ts/norm(ts))

# second derivative
function ∂²(curve::NurbsCurve,s,tᵢ)
    ForwardDiff.derivative(s->tangent(curve,s,tᵢ),s)
end
normal(curve::NurbsCurve,s::T,t) where T = (ns=∂²(curve,s,t); ns/norm(ns.+eps(T)))

# strains
function ε(curve::NurbsCurve,q::NurbsCurve,s::T,t) where T
    R(q(s,t))*∂(curve,s,t)-SA{T}[0,0,1]
end

# curvature, only applicable to quarterion curves
function κ(q::NurbsCurve{4},s,t)
    ∂R(q,s,t)*R(q(s,t))
end

# some test
cps = SA{T}[0 0.5 1 1.5
            0  0  0  0
            0  0  0  0]
cps_ϵ = SA{T}[0 0.51 1.02 1.53
               0  0  0  0
               0  0  0  0]
weights = SA{T}[1.,1.,1.,1.0]
knots =   SA{T}[0,0,0,0.5,1,1,1]
qs = SA{T}[0 0 0 0
           0 0 0 0
           0 0 0 0
           1 1 1 1] # straight

# make different types of bodies
curve = NurbsCurve(cps,knots,weights);
curve_ϵ = NurbsCurve(cps_ϵ,knots,weights);
quarterions = NurbsCurve(qs,knots,weights);

# straight line is straight
@assert all(∂(curve,0.,0.) .≈ [2,0,0])
@assert all(tangent(curve,0.0,0.0) .≈ [1,0,0])
@assert all(∂²(curve,0.,0.) .≈ [0,0,0])
@assert all(normal(curve,0.,0.) .≈ [0,0,0]) # what does that mean?
# strains
@assert all(ε(curve,quarterions,0.,0.) .≈ [2,0,-1])
@assert all(ε(curve,quarterions,0.5,0.) .≈ [1,0,-1])
@assert all(ε(curve,quarterions,1.,0.) .≈ [2,0,-1])
@assert all(κ(quarterions,0.,0.) .≈ [0 0 0; 0 0 0; 0 0 0])
@assert all(κ(quarterions,.5,0.) .≈ [0 0 0; 0 0 0; 0 0 0])
@assert all(κ(quarterions,1.,0.) .≈ [0 0 0; 0 0 0; 0 0 0])

"""
    Greville(curve::NurbsCurve{d,p};k=0) where {d,p}

Compute the Greville abscissae, defined to be the mean location of k+1 consecutive
knots in the knot vector for each basis spline function of order p.
"""
function Greville(curve::NurbsCurve{d,p};k=0) where {d,p}
    N = size(curve.pnts,2)
    SVector{N}([sum(@views(curve.knots[i+1+k:i+p]))/(p-k) for i ∈ 1:N-k])
end
@btime Greville(curve)
# Greville abscissae of knot vector [0,0,0,1,2,3,4,4,4]
curve_ = NurbsCurve(SA[0:0.2:1...; 0 0 0 0 0 0],SA[0,0,0,1,2,3,4,4,4],
                   SA[1,1,1,1,1,1])
@assert all(Greville(curve_) .≈ [0,1/2,3/2,5/2,7/2,4])

#@TODO change to Bishop frame
function DarbouxFrame(curve::NurbsCurve{3,d},s::T,t) where {d,T}
    d₃ = tangent(curve,s,t)
    d₁ = SA[0.0,1.0,0.0]# normal(curve,s,t)
    d₂ = d₃ × d₁
    return SA[d₁[1] d₁[2] d₁[3]
              d₂[1] d₂[2] d₂[3]
              d₃[1] d₃[2] d₃[3]]
end
DarbouxFrame(curve,0.,0.)


"""
    initial_rotation(curve::NurbsCurve)

Compute the initial rotation of the curve in terms of unit quarterions
"""
function initial_rotation(curve::NurbsCurve{D,p}) where {D,p}
    ξ = Greville(curve)
    T = eltype(curve.pnts)
    # for each rotation matrix, find the quaternion
    n = length(ξ)
    q = zeros(T,4,n)
    for i ∈ 1:n
        q[:,i] .= rot2quat(DarbouxFrame(curve,ξ[i],0.0))
    end
    # construct system and solve
    A = zeros(T,n,n);
    for i ∈ 1:n, k ∈ 1:n
        A[i,k] = ifelse(abs(i-k)≥p,0.0,ParametricBodies.Bd(curve.knots,ξ[i],k,Val(p)))
    end
    q = SMatrix{4,n,T}((A\q')') # bit ugly, but it works
    # build NurbsCurve and return it
    NurbsCurve(q,curve.knots,ones(T,size(q,2)))
end

q_crv = initial_rotation(curve)

struct CosseratRod{T,M<:AbstractArray{T},Sf<:AbstractArray{T},Vf<:AbstractArray{T}} <: AbstractParametricBody
    curve :: NurbsCurve
    quarterions :: NurbsCurve
    C :: M
    D :: M
    τ :: Sf
    resid :: Vf
    # how do we store the initial stains? NurbsCurve?
    function CosseratRod(curve::NurbsCurve{N,p,S},E=1,ν=1,A=1,I₁=1,I₂=1;J=I₁+I₂) where {N,p,S<:AbstractArray{T}}
        quarterions = initial_rotation(curve)
        # intrinsic material properties
        k₁,G = 5/6,E/(2+2ν)
        C = SA{T}[k₁*G*A k₁*G*A E*A] # matrix are diagonal
        D = SA{T}[E*I₁ E*I₂ G*J]
        τ = Greville(curve)
        r = zeros(T,3,length(τ))
        new{T,typeof(C),typeof(τ),typeof(r)}(curve,quarterions,C,D,τ,r)
    end
end

Rod = CosseratRod(curve)

# """
#     residual!

# Compute the residual vector of the linear and angular momentum equation of a Cosserat rod
# """
# function residual!()
#     nothing #yet!
# end

# function f(q, Rs)
#     return norm(R(q) .- Rs)
# end

println("Done!")