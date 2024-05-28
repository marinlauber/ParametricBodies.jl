# ParametricBodies

Tutorial video [![tutorial video link](https://img.youtube.com/vi/6PmJJKVOfvc/hqdefault.jpg)](https://www.youtube.com/watch?v=6PmJJKVOfvc)


This is a preliminary (unregistered) pacakge to enable working with parametrically defined shapes in [WaterLily](https://github.com/weymouth/WaterLily.jl). It defines two types, a [ParametricBody](https://github.com/weymouth/ParametricBodies.jl/blob/ec16d7efb5964c2200da65c71e643d7fbaf064c2/src/ParametricBodies.jl#L35) to hold the shape definition and shape interrogation methods, and a [HashedLocator](https://github.com/weymouth/ParametricBodies.jl/blob/ec16d7efb5964c2200da65c71e643d7fbaf064c2/src/HashedLocators.jl#L33) to robustly locate the closest point on the shape. Many of the methods are currently specific to curves (defined by only one parameter) but could be extended to surfaces fairly directly.

Until this package matures and is registered, you need to either [add it via github](https://pkgdocs.julialang.org/v1/managing-packages/#Adding-unregistered-packages) 
```
] add https://github.com/weymouth/ParametricBodies.jl
```
or download the github repo and then [activate the environment](https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project)
```
shell> git clone https://github.com/weymouth/ParametricBodies.jl
Cloning into 'ParametricBodies.jl'...
...
] activate ParametricBodies
] instantiate
```

### Usage

`ParametricBodies.jl` allows to define two types of bodies: `ParametricBody` and `DynamicBody`.

##### Parametric Bodies

A `ParametricBody` can be define from a parametric function repsentation of that shape. For example, a circle can be defined as
```julia
circle(s,t) = SA[sin(2π*s),cos(2π*s)] # a circle
Body = ParametricBody(circle, (0,1))
```
The interval `(0,1)` gives the interval in which the parametric curve is defined

`ParametricBody` can be made dynamic by specifying a mapping to the `ParametricBody` type
```julia
heave(x,t) = SA[0.,sin(2π*t)]
Body = ParametricBody(circle, (0,1); map=heave)
```


##### Dynamic Bodies

`DynamicBodies` can be constructed from `NurbsCurves` and `BSplineCurves` and allow the user to adjust (move) the curve's control points during the simulations. Internally, `BSplineCurves` are `NurbsCurves` and the `DynamicBody` is constructed from the `NurbsCurve` and a `NurbsLocator` is created to allow for a fast and robust evaluation of the location of the closest point on the curve.

Dynamic adjustement of control points requires creating the curves' control point with a `MMatrix` from the `StaticArrays` package as such

```julia
cps = MMatrix(SA[1 0 -1; 0 0 0])
spline = BSplineCurve(cps; degree=2)
Body = DynamicBody(spline,(0,1))
```
Updating the control point is can be simply done by passing the new control point position and a time step (used to compute the NURBS' velocity)
```julia
update!(DynamicBody, new_cps, Δt)
```

##### Overloading signed distance function

By default, `DynamicBody` uses a signed distance function to the parametric curve. This might not be the desired behaviour and can be overloaded by defining a new `dist` function. For example, to overload the signed distance function a spline with a thickness of `thk` one can do
```julia
new_dist(p,n) = √(p'*p) - thk/2
DynamicBody(spline,(0,1);dist=new_dist)
```
This is demonstrated in [examples/TwoD_nurbs.jl](./example/TwoD_nurbs.jl).


### Spanwise Periodic ParametricBodies

We can combine the periodic capability of `WaterLily` with a periodic `ParametricBody` defined using a 2D curve that is extruded in one of the 3 Cartesian direction. This can be very simply setup using the `perdir` argument in the `ParametricBody` constructor. For example, to define a periodic body in the `z` direction, we can do
```julia
circle(s,t) = SA[sin(2π*s),cos(2π*s)]
map(x,t) = x-SA[4L,4L]
body = ParametricBody(circle,(0,1);perdir=(3,),map)
sim = Simulation((10L,8L,L),(U,0,0),L;ν=U*L/Re,body,perdir=(3,))
```
where we have used the same argument that would be used to make the `WaterLily` simulation periodic in the z-direction. This can of course be combined with a `map` function to make the body dynamic. but the important point here to observe is that the maping must also be periodic in the `z` direction and the __`map` function must return a 2D vector__. This is demonstrated in [examples/ThreeD_tandem_airfoil_spanperiodic.jl](./example/ThreeD_tandem_airfoil_spanperiodic.jl).
