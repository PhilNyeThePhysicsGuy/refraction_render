from refraction_render.renderers import Scene,Renderer_35mm,Renderer_Composite
from refraction_render.calcs import CurveCalc,FlatCalc
from refraction_render.misc import mi_to_m,ft_to_m
from pyproj import Geod
from PIL import Image
import numpy as np
import os
import cProfile


def cfunc(d,h,n_ref,h_max,d_min):
    # this is a function which should give the color of the pixels on the
    # rendered topographical data. h_max is the maximum value of the elevation
    # which for the isle of man is 621 meters, d_min is roughly the minimum distance
    # of land away from the observer, which is roughly 50 km. 
    ng = 100+(255-100)*(d/d_min)**(-4)
    nr = ng*(1-h/h_max)

    # this line of code is used to generate the shading of the elevation.
    # n_ref are a set of unit vector pointing in the direction of the reflected 
    # ray bounding off of the surface if the lighting is above the object the closer 
    # the reflected ray is to pointing completly vertical the more illuminated it 
    # should appear. In this way we can scale the whole rgb color by a dimming factor
    # that takes into account this angle. Note we do not take the dimming to 0, 
    # otherwise the shading would be black.
    dimming = 1-0.5*np.abs(n_ref[1,:])

    return np.stack(np.broadcast_arrays(dimming*nr,dimming*ng,0),axis=-1)

def T_prof(h):
    e1 = np.exp(h/1.5)
    e2 = np.exp(h/0.1)
    return (2/(1+e1))*0.1+(2/(1+e2))*0.15

# create calculator
calc_args = dict(T0=8.3,P0=103000,T_prof=T_prof)
calc = CurveCalc(**calc_args)

# load topographical data
data = np.array(Image.open("n54_w005_1arc_v3.tif"))
n_lat,n_lon =  data.shape

lats = np.linspace(54,55,n_lat) # get latitudes of raster
lons = np.linspace(-5,-4,n_lon) # get longitudes of raster
# data must be flipped row whys so that latitude grid is strictly increasing
data = data[::-1,:].copy() 

d_max = mi_to_m(40)
h_obs,lat_obs, lon_obs = 35, 54.487375, -3.599760
h_lighthouse,lat_lighthouse, lon_lighthouse = 43.7, 54.295668, -4.309418

s = Scene()
image_path = "MG_lighthouse_model.png"
s.add_image(image_path,(h_lighthouse,lat_lighthouse,lon_lighthouse),dimensions=(-1,23))
s.add_elevation_model(lats,lons,data)


renderer = Renderer_35mm(calc,h_obs,lat_obs,lon_obs,(lat_lighthouse,lon_lighthouse),
	d_max,vert_res=2000,focal_length=2000)

renderer.render_scene(s,'lighthouse_render.png',cfunc=cfunc,cfunc_args=(621,50000),disp=True)

renderer = Renderer_Composite(calc,h_obs,lat_obs,lon_obs,d_max,vert_res=500,focal_length=2000)

heading_min = 235
heading_max = 259
renderer.render_scene(s,"Isle_of_Man_composite.png",heading_min,heading_max,
	cfunc=cfunc,cfunc_args=(621,50000),disp=True)








