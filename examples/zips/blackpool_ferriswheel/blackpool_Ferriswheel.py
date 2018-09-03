from refraction_render.renderers import Scene,Renderer_35mm
from refraction_render.calcs import CurveCalc,FlatCalc
from refraction_render.misc import mi_to_m,ft_to_m
from pyproj import Geod
import numpy as np
import matplotlib.pyplot as plt
import os


def smooth_f(x,a=1):
	return np.abs((x+a*np.logaddexp(x/a,-x/a))/2.0)

def T_prof(h):
	e=np.nan_to_num(np.exp(-smooth_f(h/4)**0.5))
	return -0.3*e

# Temperature data at time of observation
calc = CurveCalc(T0=16,P0=102000,T_prof=T_prof)


#plotting temperature profile
h = np.linspace(0,50)
plt.plot(calc.atm_model.T(h),h)
plt.xlabel("Temperature (Celsius)")
plt.ylabel("Height (meters)")
plt.savefig("T_prof.png",bbox_inches="tight",dpi=500)
plt.clf()

plt.plot(calc.atm_model.rho(h),h)
plt.xlabel("Density (kg/m$^3$)")
plt.ylabel("Height (meters)")
plt.savefig("rho_prof.png",bbox_inches="tight",dpi=500)
plt.clf()


h_obs, lat_obs, lon_obs = ft_to_m(10)  ,54.045708, -3.201814
lat_fw, lon_fw = 53.810927, -3.057650

bg_color = np.array([0,0,0],dtype=np.uint8)
sf_color = np.array([20, 5, 55],dtype=np.uint8)


renderer = Renderer_35mm(calc,h_obs,lat_obs,lon_obs,(lat_fw,lon_fw),
	mi_to_m(30),vert_res=1080,focal_length=2000)


s = Scene()
s.add_image("ferriswheel_lights.png",(ft_to_m(33),lat_fw,lon_fw),dimensions=(-1,ft_to_m(108)))
renderer.render_scene(s,"ferris_wheel.png".format(h_obs),
	background_color=bg_color,surface_color=sf_color)

