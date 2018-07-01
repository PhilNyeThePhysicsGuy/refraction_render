from ..calcs import CurveCalc,FlatCalc

from numba import njit,guvectorize
from PIL import Image
from pyproj import Geod
from six import iteritems
import scipy.interpolate as interp
import numpy as np

__all__=["Scene","Renderer_35mm"]


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
	for i in range(a.shape[0]):
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

@njit
def _update_png_data(k,l,png_data,i,j,img_png_data):
	n = k.shape[0]
	m = l.shape[0]

	for a in range(n):
		ii = i[a]
		kk = k[a]
		for b in range(m):
			if img_png_data[ii,j[b],3] > 0:
				png_data[kk,l[b],:] = img_png_data[ii,j[b],:3]

@njit
def _ray_crossing(rs,heights,inds,water,land,sky):
	n_v = rs.shape[0]
	m = rs.shape[1]

	for i in range(n_v):
		water[i] = False
		land[i] = False
		sky[i] = False
		inds[i] = -1
		hit = False
		for j in range(m):
			if rs[i,j] <= heights[j]:
				hit = True
				if heights[j] > 0.01:
					land[i] = True
					inds[i] = j
				else:
					water[i] = True
					inds[i] = j

				break

		sky[i] = not hit


def _check_gps(lat,lon):
	if np.any(lat < -90) or np.any(lat > 90):
		raise ValueError("latitude must be between -90 and 90")
	if np.any(lon < -180) or np.any(lon > 180):
		raise ValueError("lonitude must be between -180 and 180")


def _defualt_cfunc(d,heights):
	ng = 150
	# nr = ng*(1-heights/(heights.max()+1))
	return np.stack(np.broadcast_arrays(0,ng,0),axis=-1)


def _render_images(png_data,rs,ds,h_angles,img_datas,ray_heights):
	n_z = rs.shape[1]

	ind = np.zeros(rs.shape[0],dtype=np.int)
	v_mask = np.zeros(rs.shape[0],dtype=np.bool)
	h_mask = np.zeros(h_angles.shape[0],dtype=np.bool)
	_get_water(rs,ind)

	for img_png_data,h_px,v_px,d in img_datas:
		rh = ray_heights[d]

		_get_bounds(h_angles,h_px[0],h_px[-1],h_mask)
		_get_vertical_mask(rh,v_px[0],v_px[-1],ind,n_z,ds,d,v_mask)

		if np.any(v_mask):
			k = np.argwhere(h_mask).ravel()
			l = np.argwhere(v_mask).ravel()
			i = np.searchsorted(h_px,h_angles[k])
			j = np.searchsorted(v_px,rh[l])

			_update_png_data(k,l,png_data,i,j,img_png_data)


def _render_terrain(png_data,lat_obs,lon_obs,rs,ds,h_angles,terrain,cfunc,cfunc_args,surface_color,background_color):
	n_v = rs.shape[0]

	water = np.zeros(n_v,dtype=np.bool)
	land = np.zeros(n_v,dtype=np.bool)
	sky = np.zeros(n_v,dtype=np.bool)
	inds = np.zeros(n_v,dtype=np.int32)

	if terrain.has_data: # render land model first

		height = np.zeros_like(ds)

		for i,h_angle in enumerate(h_angles):

			heights = terrain.get_terrain(lat_obs,lon_obs,h_angle,ds)

			_ray_crossing(rs,heights,inds,water,land,sky)
			png_data[i,water,:] = surface_color
			png_data[i,sky,:] = background_color

			if np.any(land):
				land_inds = inds[land]
				png_data[i,land,:] = cfunc(ds[land_inds],heights[land_inds],*cfunc_args)

	else: # if no terrain to render, just render sky
		n_z = rs.shape[1]
		_get_water(rs,inds)

		np.less(inds,n_z,out=water)
		png_data[:,water,:] = surface_color
		np.logical_not(water,out=water)
		png_data[:,water,:] = background_color


class Renderer_35mm(object):
	"""

	"""
	def __init__(self,calc,h_obs,lat_obs,lon_obs,direction,max_distance,
				 distance_res=10,vert_obs_angle=0.0,vert_res=1000,
				 focal_length=2000,atol=1.1e-3,rtol=1.1e-5):

		"""

		"""

		if isinstance(calc,CurveCalc):
			self._sphere=True
			self._R0 = calc.R0
		elif isinstance(calc,FlatCalc):
			self._sphere=False
			self._R0 = 0.0
		else:
			raise ValueError("calc must be an instance of CurveCalc or FlatCalc.")

		self._lat_obs = float(lat_obs)
		self._lon_obs = float(lon_obs)
		self._h_obs = float(h_obs)
		_check_gps(self._lat_obs,self._lon_obs)

		self._calc = calc
		self._geod = Geod(ellps="sphere")

		if type(direction) is tuple:
			if len(direction) != 2:
				raise ValueError("direction must be either heading or tuple containing latitude and lonitude respectively.")

			lat_dir,lon_dir = direction
			_check_gps(lat_dir,lon_dir)
			f_az,b_az,dist = self._geod.inv(lon_obs,lat_obs,lon_dir,lat_dir)

		else:
			f_az = float(direction)

		f_az = f_az%360
		vert_res = int(vert_res)
		horz_res = int(vert_res*1.5)
		self._vert_res = vert_res
		self._horz_res = horz_res
		self._focal_length = focal_length

		y_grid = np.linspace(-12,12,vert_res)
		x_grid = np.linspace(-18,18,horz_res)
		self._v_angles = np.rad2deg(np.arctan(y_grid/focal_length))+vert_obs_angle
		self._h_angles = np.rad2deg(np.arctan(x_grid/focal_length))+f_az

		self._ds = np.arange(0,max_distance,distance_res)
		if self._sphere:
			self._sols = calc.solve_ivp(max_distance,h_obs,alpha=self._v_angles,atol=atol,rtol=rtol,dense_output=True)
		else:
			self._sols = calc.solve_ivp(max_distance,h_obs,alpha=self._v_angles,atol=atol,rtol=rtol,dense_output=True)

		if self._sphere:
			sigmas = np.pi/2+self._ds/self._R0
			self._rs = (self._sols.sol(sigmas)[:vert_res]-self._R0).copy()
		else:
			self._rs = self._sols.sol(self._ds)[:vert_res].copy()

	def change_direction(self,direction):
		"""

		"""
		if type(direction) is tuple:
			if len(direction) != 2:
				raise ValueError("direction must be either heading or tuple containing latitude and lonitude respectively.")

			lat_dir,lon_dir = direction
			_check_gps(lat_dir,lon_dir)
			f_az,b_az,dist = self._geod.inv(lon_obs,lat_obs,lon_dir,lat_dir)

		else:
			f_az = float(direction)

		f_az = f_az%360

		x_grid = np.linspace(-18,18,self._horz_res)
		self._h_angles = np.rad2deg(np.arctan(x_grid/self._focal_length))+f_az


	def render_scene(self,scene,image_name,surface_color=None,background_color=None,cfunc=_defualt_cfunc,cfunc_args=()):
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

			for lat,lon,vert_pixel_pos,horz_pixel_pos in image_data:
				v_px = vert_pixel_pos
				f_az,b_az,dist = self._geod.inv(self._lon_obs,self._lat_obs,lon,lat)

				h_px=np.arctan(horz_pixel_pos/dist)
				np.rad2deg(h_px,out=h_px)
				h_px += (f_az % 360)
				np.mod(h_px,360,out=h_px)

				if dist not in ray_heights:
					if self._sphere:
						s = np.pi/2+dist/self._R0
						ray_heights[dist] = (self._sols.sol(s)[:self._vert_res]-self._R0).copy()
					else:
						ray_heights[dist] = self._sols.sol(dist)[:self._vert_res].copy()


				img_datas.append((im_data,h_px,v_px,dist))

		img_datas.sort(key=lambda x:-x[-1])

		d_min = np.inf
		h_max = 0
		
		land_model = scene._land_model

		png_data = np.empty((len(self._h_angles),self._rs.shape[0],3),dtype=np.uint8)
		png_data[...] = 0

		_render_terrain(png_data,self._lat_obs,self._lon_obs,self._rs,self._ds,self._h_angles,land_model,cfunc,cfunc_args,surface_color,background_color)
		_render_images(png_data,self._rs,self._ds,self._h_angles,img_datas,ray_heights)

		png_data = png_data.transpose((1,0,2))
		png_data = png_data[::-1,:,:]
		im = Image.fromarray(png_data,mode="RGB")
		im.save(image_name)


class Scene(object):
	def __init__(self,ellps='sphere'):
		"""
		Simple wrapper which keeps track of data used to render an image.

		Parameters
		----------
		ellps: str, optional
			String representing which pyproj datum to use for the elevation data being interpolated.

		"""
		self._land_model = _land_model()
		self._image_dict = {}

	def add_elevation_model(self,lats,lons,elevation):
		self._land_model.add_elevation_data(lats,lons,elevation)

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
			lat,lon = image_pos
			h = 0
		elif len(image_pos) == 3:
			h,lat,lon = image_pos
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

		self._image_dict[image][1].append((lat,lon,vert_pixel_pos,horz_pixel_pos))

		



class _land_model(object):
	"""
	This class is used to grab slices of the tarrain alon great circle
	slices of the earth. 
	"""
	def __init__(self,ellps='sphere'):
		"""
		Simple wrapper which does the bilinear interpolation of terrain data.

		Parameters
		----------
		ellps: str, optional
			String representing which pyproj datum to use for the data being interpolated.

		"""
		self._geod = Geod(ellps=ellps)
		self._terrain_list = []
		self._d_min = np.inf
		self._d_max = 0
		self._h_max = 0

	def get_terrain(self,lat,lon,heading,dist):
		"""
		Sample terrain along geodesic defined by an intial latitude and longitude point along a heading at different distances.

		Parameters
		----------
		lat: array_like
			latitude(s) which starting point(s) begin at
		lon: array_like
			longitude(s) which starting point(s) begin at
		heading: array_like
			heading(s) which determine which direction to go on the spheriod
		dist: array_like
			distance(s) in meters of how far to go along the heading

		Notes
		-----
		This function follows NumPy's broadcasting rules for handing arguments of different shapes.

		"""
		lon,lat,az = self._geod.fwd(*np.broadcast_arrays(lon,lat,heading,dist))

		return self.__call__(lat,lon)

	def __call__(self,lat,lon):
		lat,lon = np.broadcast_arrays(lat,lon)
		heights = np.zeros_like(lon)

		coors = np.stack((lat,lon),axis=-1)
		for terrain in self._terrain_list:
			heights += terrain(coors)

		return heights

	@property
	def has_data(self):
		return len(self._terrain_list) > 0

	def add_elevation_data(self,lats,lons,elevation):
		""" 
		Add terrain data to the interpolated model.

		Parameters
		----------
		lats: ndarray with shape (n,)
			latitudes used in grid.

		lons: ndarray with shape (m,)
			longitudes used in grid.

		data: ndarray with shape (n,m)
			elevation data at each grid defined by lats and lons.

		"""
		self._terrain_list.append(interp.RegularGridInterpolator((lats,lons),elevation,bounds_error=False,fill_value=0.0))

