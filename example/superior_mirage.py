from refraction_render.renderers import Scene,Renderer_35mm
from refraction_render.calcs import CurveCalc,FlatCalc
from refraction_render.misc import mi_to_m,ft_to_m

from pyproj import Geod
import numpy as np
import os


def T_prof(h):
	return 1.2*np.exp(-((h-23)/3)**2)

calc_args = dict(T_prof=T_prof)
calc = CurveCalc(**calc_args)


geod = Geod(ellps="sphere")
s = Scene()

dist_boat = mi_to_m(10)
heading_boat = 270


h_obs, lat_obs, lon_obs = 20, 54.487375, -3.599760
lon_1,lat_1,b_az = geod.fwd(lon_obs,lat_obs,270,dist_boat)

renderer = Renderer_35mm(calc,h_obs,lat_obs,lon_obs,heading_boat,mi_to_m(20),
							vert_res=4000,focal_length=2000,vert_obs_angle=-0.05)

image1_path = "cargo_2.png"
s.add_image(image1_path,(-0.5,lat_1,lon_1),dimensions=(ft_to_m(500),-1))
renderer.render_scene(s,"boat_superior_mirage.png")

