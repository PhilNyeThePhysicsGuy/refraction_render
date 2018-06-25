from ..calcs import CurveCalc,FlatCalc

from numba import njit
from PIL import Image
from pyproj import Geod
from six import iteritems
import numpy as np

__all__=["Scene","Renderer_35mm"]

def _check_gps(theta,phi):
	if np.any(theta < -90) or np.any(theta > 90):
		raise ValueError("latitude must be between -90 and 90")
	if np.any(phi < -180) or np.any(phi > 180):
		raise ValueError("longitude must be between -180 and 180")

@njit
def _get_water(rs,ind):
	n_v = rs.shape[0]
	m = rs.shape[1]
	ind[:] = m
	for i in range(n_v):
		for j in range(m):
			if rs[i,j] <= 0:
				ind[i] = j
				break

@njit
def _get_bounds(a,a_min,a_max,mask):
	n = a.shape[0]
	mask[:] = False

	for i in range(n):
		mask[i] = a[i] >= a_min and a[i] < a_max

@njit
def _get_vertical_mask(rh,h_min,h_max,ind,n_z,ds,d,mask):
	n = rh.shape[0]

	for i in range(n):
		if rh[i] >= h_min and rh[i] < h_max:
			if ind[i] < n_z:
				if ds[ind[i]] < d:
					mask[i] = False
				else:
					mask[i] = True
			else:
				mask[i] = True
		else:
			mask[i] = False


def _render_images(rs,ds,h_angles,img_datas,ray_heights,surface_color,background_color):
	png_data = np.empty((len(h_angles),rs.shape[0],3),dtype=np.uint8)
	png_data[...] = 0

	n_z = rs.shape[1]

	ind = np.zeros(rs.shape[0],dtype=np.int)
	v_mask = np.zeros(rs.shape[0],dtype=np.bool)
	h_mask = np.zeros(h_angles.shape[0],dtype=np.bool)
	_get_water(rs,ind)

	w_mask = ind < n_z
	png_data[:,w_mask,:] = surface_color
	np.logical_not(w_mask,out=w_mask)
	png_data[:,w_mask,:] = background_color

	for img_png,h_px,v_px,d in img_datas:
		rh = ray_heights[d]

		_get_bounds(h_angles,h_px[0],h_px[-1],h_mask)
		_get_vertical_mask(rh,v_px[0],v_px[-1],ind,n_z,ds,d,v_mask)

		if np.any(v_mask):
			k = np.argwhere(h_mask).ravel()
			l = np.argwhere(v_mask).ravel()

			kk = np.kron(np.ones_like(l),k)
			ll = np.kron(l,np.ones_like(k))

			i = np.searchsorted(h_px,h_angles[h_mask])
			j = np.searchsorted(v_px,rh[v_mask])

			ii = np.kron(np.ones_like(j),i)
			jj = np.kron(j,np.ones_like(i))
			
			mm = img_png[ii,jj,3] > 0

			if np.any(mm):
				png_data[kk[mm],ll[mm],:] = img_png[ii[mm],jj[mm],:3]


	return png_data


class Renderer_35mm(object):
	def __init__(self,calc,h_i,theta_i,phi_i,direction,max_distance,distance_res=10,vert_camera_tilt=0.0,vert_res=1000,focal_length=2000,R0=6370997.0,rectilinear=True):
		if isinstance(calc,CurveCalc):
			self._sphere=True
			self._R0 = R0
		elif isinstance(calc,FlatCalc):
			self._sphere=False
			self._R0 = 0.0
		else:
			raise ValueError("calc must be an instance of CurveCalc or FlatCalc.")

		self._theta_i = float(theta_i)
		self._phi_i = float(phi_i)
		self._h_i = float(h_i)
		_check_gps(self._theta_i,self._phi_i)

		self._calc = calc
		self._geod = Geod(ellps="sphere")

		if type(direction) is tuple:
			if len(direction) != 2:
				raise ValueError("direction must be either heading or tuple containing latitude and Longitude respectively.")

			theta_dir,phi_dir = direction
			_check_gps(theta_dir,phi_dir)
			f_az,b_az,dist = self._geod.inv(phi_i,theta_i,phi_dir,theta_dir)

		else:
			f_az = float(direction)

		f_az = f_az%360
		vert_res = int(vert_res)
		horz_res = int(vert_res*1.5)
		self._vert_res = vert_res
		self._horz_res = horz_res

		if rectilinear:
			y_grid = np.linspace(-12,12,vert_res)
			x_grid = np.linspace(-18,18,horz_res)
			self._v_angles = np.rad2deg(np.arctan(y_grid/focal_length))+vert_camera_tilt
			self._h_angles = np.rad2deg(np.arctan(x_grid/focal_length))+f_az
		else:
			a_v = np.rad2deg(2*np.arctan(12.0/focal_length))
			a_h = np.rad2deg(2*np.arctan(18.0/focal_length))
			self._v_angles = np.linspace(-a_v/2,a_v/2,vert_res)+vert_camera_tilt
			self._h_angles = np.linspace(-a_h/2,a_h/2,horz_res)+f_az

		self._ds = np.arange(0,max_distance,distance_res)
		if self._sphere:
			self._sols = calc.solve_ivp(max_distance,h_i,R0=self._R0,alpha=self._v_angles,atol=1.1e-5,rtol=1.1e-7)
		else:
			self._sols = calc.solve_ivp(max_distance,h_i,alpha=self._v_angles,atol=1.1e-5,rtol=1.1e-7)

		if self._sphere:
			sigmas = np.pi/2+self._ds/self._R0
			self._rs = (self._sols.sol(sigmas)[:vert_res]-self._R0).copy()
		else:
			self._rs = self._sols.sol(self._ds)[:vert_res].copy()



	def render_scene(self,scene,image_name,surface_color=None,background_color=None):
		if surface_color is None:
			surface_color = np.array([0,80,120])

		if background_color is None:
			background_color = np.array([135,206,250])

		if not isinstance(scene,Scene):
			raise ValueError("scene must be a scene type object")

		img_datas = []
		ray_heights = {}

		for image,(im,image_data) in iteritems(scene._image_dict):
			im_data = np.array(im)
			im_data = im_data[::-1,:,:].transpose((1,0,2)).copy()

			for theta,phi,vert_pixel_pos,horz_pixel_pos in image_data:
				v_px = vert_pixel_pos
				f_az,b_az,dist = self._geod.inv(self._phi_i,self._theta_i,phi,theta)
				f_az = f_az%360

				h_px=np.arctan(horz_pixel_pos/dist)
				np.rad2deg(h_px,out=h_px)
				h_px += (f_az)
				np.mod(h_px,360,out=h_px)

				if dist not in ray_heights:
					if self._sphere:
						s = np.pi/2+dist/self._R0
						ray_heights[dist] = (self._sols.sol(s)[:self._vert_res]-self._R0).copy()
					else:
						ray_heights[dist] = self._sols.sol(dist)[:self._vert_res].copy()


				img_datas.append((im_data,h_px,v_px,dist))

		img_datas.sort(key=lambda x:-x[-1])

		png_data = _render_images(self._rs,self._ds,self._h_angles,img_datas,ray_heights,surface_color,background_color)

		png_data = png_data.transpose((1,0,2))
		png_data = png_data[::-1,:,:]
		im = Image.fromarray(png_data,mode="RGB")
		im.save(image_name)


class Scene(object):
	def __init__(self):
		self._image_dict = {}

	def add_image(self,image,image_pos,dimensions=None,pixel_size=None):

		if type(image) is str:
			if image in self._image_dict:
				im = self._image_dict[image][0]
			else:
				im = Image.open(image)
				self._image_dict[image] = (im,[])

		px_width,px_height = im.size

		aspect = float(px_height)/px_width

		if len(image_pos) == 2:
			theta,phi = image_pos
			h = 0
		elif len(image_pos) == 3:
			h,theta,phi = image_pos
		else:
			raise ValueError("expecting image_pos to contain gps coordinates and optionally height of image.")


		if dimensions is None and pixel_size is None:
			raise ValueError("image requires dimensions or pixel_size.")

		if dimensions is not None and pixel_size is not None:
			raise ValueError("only use either dimensions or pixel_size not both.")

		if dimensions is not None:
			try:
				width,height = dimensions
			except ValueError:
				raise ValueError("dimensions must contain only length and width of object.")

			if width == -1 and height == -1:
				raise ValueError("at least one height must be specifided to deduce size.")

			if width == -1:
				width = height/aspect

			if height == -1:
				height = width*aspect

			vert_pixel_pos = h + np.linspace(0,height,px_height)
			horz_pixel_pos = np.linspace(-width/2,width/2,px_width)



		if pixel_size is not None:
			try:
				pixel_size = float(pixel_size)
			except ValueError:
				raise ValueError("pixel size must be a scalar value corresponding to the physical size of a side of a pixel in image.")

			vert_pixel_pos = h + np.arange(px_height)*pixel_size
			horz_pixel_pos = pixel_size*np.arange(-(px_width//2+px_width%2),px_width//2,1)

		self._image_dict[image][1].append((theta,phi,vert_pixel_pos,horz_pixel_pos))

		


