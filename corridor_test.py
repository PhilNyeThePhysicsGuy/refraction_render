from refraction_render.calcs import atmospheric_corridor,FermatEquationsCurve,FermatEquationsEuclid,Calc
from refraction_render.renderers import ray_diagram
from refraction_render.misc import km_to_m
import matplotlib.pyplot as plt
from pyproj import Geod
import numpy as np


def T_prof(h,d,d0,D):
	return (0.0098*h+15)-100*np.exp(-np.logaddexp(0,(-(d-d0)/D)))*np.exp(-np.logaddexp(0,h/0.5))


R0 = 6370997.0
D = 0.001
d0 = 100
d_max = 200
h_obs = 1.0

d1 = np.linspace(0,d0-10*D,5)
d2 = np.linspace(d0-10*D,d0+10*D,1000)
d3 = np.linspace(d0+10*D,d_max,5)
dist_vals = np.hstack((d1,d2,d3))
dist_vals = np.unique(dist_vals)

T_prof_args = (dist_vals,d0,D)

atm_model = atmospheric_corridor(T_prof,dist_vals,T_prof_args=T_prof_args)

# f_eq = FermatEquationsCurve(R0,atm_model.f)
f_eq = FermatEquationsEuclid(atm_model.f)


plt.plot(dist_vals,atm_model.rho(dist_vals,h_obs))
plt.show()
calc = Calc(f_eq)

ax = plt.gca()


angles = np.linspace(-0.01,0.01,101)
d = np.arange(0,d_max,10)
ray_diagram(ax,calc,h_obs,d,angles,style="flat")

plt.show()




