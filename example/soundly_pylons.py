import sys
sys.path.insert(0,"../")
from refraction_render.renderers import Scene,Renderer_35mm
from refraction_render.calcs import CurveCalc,FlatCalc
from pyproj import Geod
import numpy as np
import os


def T_prof(h):
	e = np.nan_to_num(np.exp(h/0.01))
	return (2/(1+e))*0.05

calc_args = dict(T_prof=T_prof)
calc = CurveCalc(**calc_args)


s = Scene()
geod = Geod(ellps="sphere")

# gps coordinates for the first two pylons
lat_1,lon_1 = 30.084791, -90.401287
lat_2,lon_2 = 30.087219, -90.400237
# getting the distance between pylongs and the heading in which 
# the rest of the pylons will follow across the lake
f_az,b_az,dist = geod.inv(lon_1,lat_1,lon_2,lat_2)
# calculating the distances (Note I got this info from google earth)
dists = np.arange(0,24820,dist)
# image path for pylon image
image_path ="pylon.png" 
# looping over distances calculating the gps position of each pylon and
# adding an image in that position
for d in dists:
	lon,lat,b_az = geod.fwd(lon_1,lat_1,f_az,d)
	s.add_image(image_path,(0,lat,lon),dimensions=(-1,23),direction=b_az)

# Soundly's position 
lat_i, lon_i = 30.077320, -90.404888 
# position of final pylon which will be centered in the frame
lat_f, lon_f = lat, lon 

# render image with wide field of view
renderer = Renderer_35mm(calc,4,lat_i,lon_i,(lat_f,lon_f),40000,vert_obs_angle=0.2,
							vert_res=5000,focal_length=400)
renderer.render_scene(s,"soundly_pylons.png")

# render image with small field of view effectively zooming in
renderer = Renderer_35mm(calc,4,lat_i,lon_i,(lat_f,lon_f),40000,vert_obs_angle=0.2,
							vert_res=5000,focal_length=2000)
renderer.render_scene(s,"soundly_pylons_zoom.png")