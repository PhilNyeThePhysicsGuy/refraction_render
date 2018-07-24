from refraction_render.renderers import Scene,Renderer_35mm,Renderer_Composite
from refraction_render.calcs import CurveCalc,FlatCalc
from refraction_render.misc import mi_to_m,ft_to_m

from pyproj import Geod
import gdal
import numpy as np
import os
import cProfile

def cfunc(d,h):
	ng = 100+(255-100)*(d/50000.0)**(-4)
	nr = ng*(1-h/621.0)
	return np.stack(np.broadcast_arrays(nr,ng,0),axis=-1)

def T_prof(h):
	e1 = np.exp(h/1.5)
	e2 = np.exp(h/0.1)
	return (2/(1+e1))*0.1+(2/(1+e2))*0.15


calc_args = dict(T0=8.3,P0=103000,T_prof=T_prof)
calc = CurveCalc(**calc_args)

raster = gdal.Open("n54_w005_1arc_v3.tif")
data = np.array(raster.ReadAsArray())
n_lat,n_lon =  data.shape

lats = np.linspace(54,55,n_lat) # get latitudes of raster
lons = np.linspace(-5,-4,n_lon) # get longitudes of raster
data = data[::-1,:].copy() # data must be flipped row whys so that latitude grid is strictly increasing

d_max = 70000
theta_i, phi_i = 54.487375, -3.599760
theta_f, phi_f = 54.295668, -4.309418

s = Scene()
image_path = os.path.join("images","MG_lighthouse_model.png")
s.add_image(image_path,(43.7,theta_f,phi_f),dimensions=(-1,23))
s.add_elevation_model(lats,lons,data)


renderer = Renderer_35mm(calc,35,theta_i,phi_i,(theta_f,phi_f),d_max,
							vert_res=2000,focal_length=2000,vert_obs_angle=0)
renderer.render_scene(s,'lighthouse_render.png',cfunc=cfunc,disp=True)

renderer = Renderer_Composite(calc,35,theta_i,phi_i,d_max,
							vert_res=500,focal_length=2000,vert_obs_angle=0)

heading_min = 235
heading_max = 259
renderer.render_scene(s,"Isle_of_Man_composite.png",heading_min,heading_max,cfunc=cfunc,disp=True)








