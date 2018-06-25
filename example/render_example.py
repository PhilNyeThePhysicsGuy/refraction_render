from refraction_render.renderers import Scene,Renderer_35mm
from refraction_render.calcs import CurveCalc,FlatCalc
from pyproj import Geod
import numpy as np
import os



def maugold_lighthouse(calc):

	d_max = 52000
	theta_i, phi_i = 54.487375, -3.599760
	theta_f, phi_f = 54.295668, -4.309418

	
	renderer = Renderer_35mm(calc,35,theta_i,phi_i,(theta_f,phi_f),d_max,
								vert_res=4000,focal_length=2000,vert_camera_tilt=0)


	s = Scene()
	image_path = os.path.join("images","MG_lighthouse.png")
	s.add_image(image_path,(43.7,theta_f,phi_f),dimensions=(-1,23))

	color = (0, 90, 40) # green background	
	renderer.render_scene(s,"IOM_lighthouse.png",background_color=color)





def ships(calc):
	geod = Geod(ellps="sphere")
	s = Scene()

	d1 = 14000
	d2 = 13000

	theta_i, phi_i = 54.487375, -3.599760
	phi_1,theta_1,b_az = geod.fwd(phi_i,theta_i,270,d1)
	phi_2,theta_2,b_az = geod.fwd(phi_i,theta_i,270+np.rad2deg(np.arctan(10/13000.0)),d2)

	renderer = Renderer_35mm(calc,10,theta_i,phi_i,270,30000,
								vert_res=4000,focal_length=4000,vert_camera_tilt=-0.07)

	image1_path = os.path.join("images","cargo2.png")
	image2_path = os.path.join("images","iStenaLine.png")
	s.add_image(image1_path,(-1.0,theta_2,phi_2),dimensions=(-1,5))
	s.add_image(image2_path,(-0.5,theta_1,phi_1),dimensions=(-1,10))
	renderer.render_scene(s,"boat_mirage.png")



def soundly_pylons(calc):
	s = Scene()
	geod = Geod(ellps="sphere")

	theta_1,phi_1 = 30.084791, -90.401287
	theta_2,phi_2 = 30.087219, -90.400237

	f_az,b_az,dist = geod.inv(phi_1,theta_1,phi_2,theta_2)
	dists = np.arange(0,24820,dist)


	image_path = os.path.join("images","pylon.png")
	for d in dists:
		phi,theta,b_az = geod.fwd(phi_1,theta_1,f_az,d)
		s.add_image(image_path,(0,theta,phi),dimensions=(-1,23))

	theta_i, phi_i = 30.077320, -90.404888
	theta_f, phi_f = 30.293719, -90.310753
	f_az,b_az,dist = geod.inv(phi_i,theta_i,phi_f,theta_f)
	f_az = f_az%360 # f_az must be between 0 and 360
	f_az = f_az + 0.3

	renderer = Renderer_35mm(calc,4,theta_i,phi_i,f_az,40000,vert_camera_tilt=0.2,
								vert_res=6000,focal_length=400)
	renderer.render_scene(s,"soundly_pylons.png")

	renderer = Renderer_35mm(calc,4,theta_i,phi_i,f_az,40000,vert_camera_tilt=0.2,
								vert_res=6000,focal_length=2000)
	renderer.render_scene(s,"soundly_pylons_zoom.png")



def T_prof_inf(h):
	e = np.nan_to_num(np.exp(h/0.01))
	return (2/(1+e))*0.05

def T_prof(h):
	e1 = np.exp(h/1.5)
	e2 = np.exp(h/0.1)
	return (2/(1+e1))*0.1+(2/(1+e2))*0.15


calc_args = dict(T0=8.3,P0=103000,T_prof=T_prof)
# calc = CurveCalc(**calc_args)
calc = FlatCalc(**calc_args)

soundly_pylons(calc)
# maugold_lighthouse(calc)

# calc_args = dict(T0=30,T_prof=T_prof_inf)
# calc = CurveCalc(**calc_args)
# calc = FlatCalc(**calc_args)
# ships(calc)
