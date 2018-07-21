from __future__ import division,print_function

from ..calcs import CurveCalc,FlatCalc

from numba import njit,guvectorize,vectorize
from PIL import Image
from pyproj import Geod
from six import iteritems
import scipy.interpolate as interp
import numpy as np

__all__=["Scene","Renderer_35mm","Renderer_Composite","land_model"]


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

@vectorize(["b1(f8,f8,f8)"])
def _get_bounds(a,a_min,a_max):
    return a >= a_min and a < a_max

@njit
def _get_vertical_mask(rh,h_min,h_max,inds,ds,d,sky,mask):
    n = rh.shape[0]

    for i in range(n):
        mask[i] = False
        if rh[i] >= h_min and rh[i] < h_max:
            if sky[i]: # if ray has hit sky (not hit anything) it has hit image
                mask[i] = True
            else:
                if ds[inds[i]] < d: # if ray hits land or water before image no change required
                    mask[i] = False
                else: 
                    mask[i] = True                

@njit
def _ray_crossing(rs,heights,inds,water,land,sky):
    n_v = rs.shape[0]
    m = rs.shape[1]
    water[:] = False
    land[:] = False
    sky[:] = False
    inds[:] = -1
    for i in range(n_v):
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

@njit
def _update_png_data(i,j,png_data,k,l,img_png_data):
    n = i.shape[0]
    m = j.shape[0]

    for a in range(n):
        ii = i[a]
        kk = k[a]
        for b in range(m):
            if img_png_data[kk,l[b],3] > 0:
                png_data[ii,j[b],:] = img_png_data[kk,l[b],:3]

@njit
def _update_png_data_slice(j,png_data,l,img_png_data):
    m = j.shape[0]

    for b in range(m):
        if img_png_data[l[b],3] > 0:
            png_data[j[b],:] = img_png_data[l[b],:3]


def _check_gps(lat,lon):
    if np.any(lat < -90) or np.any(lat > 90):
        raise ValueError("latitude must be between -90 and 90")
    if np.any(lon < -180) or np.any(lon > 180):
        raise ValueError("lonitude must be between -180 and 180")


def _defualt_cfunc(d,heights):
    ng = 150
    # nr = ng*(1-heights/(heights.max()+1))
    return np.stack(np.broadcast_arrays(0,ng,0),axis=-1)


def _render(png_data,rs,ds,h_angles,surface_color,background_color,terrain_args,image_args,disp=False):

    lat_obs,lon_obs,terrain,cfunc,cfunc_args = terrain_args
    img_datas,ray_heights = image_args

    n_v = rs.shape[0]
    n_z = rs.shape[1]

    water = np.zeros(n_v,dtype=np.bool)
    land = np.zeros(n_v,dtype=np.bool)
    sky = np.zeros(n_v,dtype=np.bool)
    inds = np.zeros(n_v,dtype=np.int32)

    v_mask = np.zeros(n_v,dtype=np.bool)
    h_mask = np.zeros(h_angles.shape[0],dtype=np.bool)

    if terrain.has_data: # render land model 

        h_mins = np.array([h_px.min() for _,h_px,_,_ in img_datas])
        h_maxs = np.array([h_px.max() for _,h_px,_,_ in img_datas])
        img_mask = np.zeros_like(h_mins,dtype=np.bool)
        h_angle_max = h_angles.max()

        for i,h_angle in enumerate(h_angles):
            if disp:
                print("{:5.5f} {:5.5f}".format(h_angle,h_angle_max))
            heights = terrain.get_terrain(lat_obs,lon_obs,h_angle,ds)

            _ray_crossing(rs,heights,inds,water,land,sky)
            png_data[i,water,:] = surface_color
            png_data[i,sky,:] = background_color

            if np.any(land):
                land_inds = inds[land]
                png_data[i,land,:] = cfunc(ds[land_inds],heights[land_inds],*cfunc_args)

            _get_bounds(h_angle,h_mins,h_maxs,out=img_mask)
            if np.any(img_mask):
                img_indx = np.argwhere(img_mask).ravel()
                for I in img_indx:
                    img_png_data,h_px,v_px,d = img_datas[I]

                    rh = ray_heights[d]
                    _get_vertical_mask(rh,v_px[0],v_px[-1],inds,ds,d,sky,v_mask)

                    j = np.argwhere(v_mask).ravel()
                    k = np.searchsorted(h_px,h_angle)
                    l = np.searchsorted(v_px,rh[j])
                    _update_png_data_slice(j,png_data[i,...],l,img_png_data[k,...])

    else: # if no terrain to render, just render sky and sphere surface

        _get_water(rs,inds)
        np.less(inds,n_z,out=water)
        png_data[:,water,:] = surface_color
        np.logical_not(water,out=sky)
        png_data[:,sky,:] = background_color
        for img_png_data,h_px,v_px,d in img_datas:
            rh = ray_heights[d]
            _get_bounds(h_angles,h_px[0],h_px[-1],out=h_mask)
            _get_vertical_mask(rh,v_px[0],v_px[-1],inds,ds,d,sky,v_mask)
            if np.any(v_mask):
                i = np.argwhere(h_mask).ravel()
                j = np.argwhere(v_mask).ravel()
                k = np.searchsorted(h_px,h_angles[i])
                l = np.searchsorted(v_px,rh[j])

                _update_png_data(i,j,png_data,k,l,img_png_data)
                

class Renderer_35mm(object):
    """
    Object used to set a camera angle and position to render a scene.
    """
    def __init__(self,calc,h_obs,lat_obs,lon_obs,direction,max_distance,
                 distance_res=10,vert_obs_angle=0.0,vert_res=1000,
                 focal_length=2000,atol=1.1e-3,rtol=1.1e-5):

        """
        Parameters
        ----------

        calc: calcs object
            object which can be used to calculate the trajectory of light rays

        h_obs: float
            height of renderer in meters

        lat_obs: float
            latitude of renderer in degrees between -90 and 90

        lon_obs: float
            longitude of renderer in degrees between -180 adn 180

        direction: tuple or float
            tuple: tuple containing a latitude and longitude point to point renderer towards.
            float: heading to point the renderer in. 

        max_distance: float
            maximum distance away from the renderer to calculate the light ray path

        distance_res: float
            distance between points where the light ray path is checked for intersection point.

        vert_obs_angle: float
            vertical tilt of the renderer. 

        vert_res: int
            resolution of pixels on the verticle

        focal_length: float
            focal length in milimeters to calculate the field of view of the camera. This sets the resolution on the horizontal.

        atol: float
            absolute tolerance of ode solver

        rtol: float
            relative tolerance of the ode solver


        """
        if isinstance(calc,CurveCalc):
            self._sphere = True
            self._R0 = calc.R0
        elif isinstance(calc,FlatCalc):
            self._sphere = False
        else:
            raise ValueError

        self._calc = calc
        self._geod = Geod(ellps="sphere")


        self._atol=atol
        self._rtol=rtol

        self._lat_obs = float(lat_obs)
        self._lon_obs = float(lon_obs)
        _check_gps(self._lat_obs,self._lon_obs)
        if type(direction) is tuple:
            if len(direction) != 2:
                raise ValueError("direction must be either heading or tuple containing latitude and lonitude respectively.")

            lat_dir,lon_dir = direction
            _check_gps(lat_dir,lon_dir)
            f_az,b_az,dist = self._geod.inv(lon_obs,lat_obs,lon_dir,lat_dir)

        else:
            f_az = float(direction)

        self._h_obs = float(h_obs)






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
        self._vfov = self._v_angles.max()-self._v_angles.min()

        self._ds = np.arange(0.0,max_distance,distance_res,dtype=np.float64)
        if self._sphere:
            sigmas = self._ds/self._R0
            sol = self._calc.solve_ivp(max_distance,h_obs,alpha=self._v_angles,atol=atol,rtol=rtol,dense_output=True)
            self._rs = (sol.sol(sigmas)[:vert_res]-self._R0)
        else:
            sol = self._calc.solve_ivp(max_distance,h_obs,alpha=self._v_angles,atol=atol,rtol=rtol,dense_output=True)
            self._rs = sol.sol(self._ds)[:vert_res].copy()

        
        self._sol = interp.interp1d(self._ds,self._rs,axis=-1,copy=False,bounds_error=True)

    @property
    def vfov(self):
        return self._vfov

    def set_location(self,lat_obs,lon_obs,direction):
        """This function can be used to change the heading of the renderer.

        Parameters
        ----------
        lat_obs: float
            new latitude of observer
        lon_obs: float
            new longitude of observer
        direction: tuple or float
            tuple: tuple containing a latitude and longitude point to point renderer towards.
            float: heading to point the renderer in. 

        """
        self._lat_obs = float(lat_obs)
        self._lon_obs = float(lon_obs)
        _check_gps(self._lat_obs,self._lon_obs)
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

    def render_scene(self,scene,image_name,surface_color=None,background_color=None,cfunc=_defualt_cfunc,cfunc_args=(),disp=False,eye_level=False):
        """This function can be used to change the heading of the renderer.

        Parameters
        ----------
        scene: Scene object
            object which contains data which the renderer can extract and render

        image_name: str
            name for the image

        background_color: array_like
            3 element array containing an color via RGB color code (numbered 0 to 255)
            default value: `[135,206,250]`

        surface_color: array_like
            3 element array containing an color via RGB color code (numbered 0 to 255)
            default value: `[0,80,120]`

        cfunc: callable
            Callable function which is used to color the elevation data. The function takes 
            in two arrays, the first is distances from the observer    the second is the elevation
            above the surface of the earth. 

        cfunc_args: array_like
            extra arguments to pass into `cfunc`

        """        
        if surface_color is None:
            surface_color = np.array([0,80,120],dtype=np.uint8)
        else:
            surface_color = np.fromiter(surface_color,dtype=np.uint8)

        if background_color is None:
            background_color = np.array([135,206,250],dtype=np.uint8)
        else:
            background_color = np.fromiter(background_color,dtype=np.uint8)

        img_datas = []
        ray_heights = {}

        for image,(im,image_data) in iteritems(scene._image_dict):
            im_data = np.array(im)
            im_data = im_data[::-1,:,:].transpose((1,0,2)).copy()

            for lat,lon,vert_pixel_pos,horz_pixel_pos,heading in image_data:
                v_px = vert_pixel_pos
                f_az,b_az,dist = self._geod.inv(self._lon_obs,self._lat_obs,lon,lat)

                if heading is None:
                    alpha = 0.0
                else:
                    b_az = b_az%360
                    heading = heading%360
                    alpha = np.deg2rad(heading - b_az)

                h_px=np.arctan(horz_pixel_pos*np.cos(alpha)/dist)
                np.rad2deg(h_px,out=h_px)
                h_px += (f_az % 360)
                np.mod(h_px,360,out=h_px)

                if dist not in ray_heights:
                    ray_heights[dist] = self._sol(dist)

                img_datas.append((im_data,h_px,v_px,dist))

        img_datas.sort(key=lambda x:-x[-1])

        land_model = scene._land_model

        png_data = np.empty((len(self._h_angles),self._rs.shape[0],3),dtype=np.uint8)
        png_data[...] = 0

        terrain_args = (self._lat_obs,self._lon_obs,land_model,cfunc,cfunc_args)
        image_args = (img_datas,ray_heights)

        _render(png_data,self._rs,self._ds,self._h_angles,surface_color,background_color,terrain_args,image_args,disp)


        if eye_level:
            i_horz = np.argwhere(np.abs(self._v_angles)<(self._vfov/400.0)).ravel()
            png_data[:,i_horz,0] = 255
            png_data[:,i_horz,1] = 100
            png_data[:,i_horz,2] = 0


        png_data = png_data.transpose((1,0,2))
        png_data = png_data[::-1,:,:]
        im = Image.fromarray(png_data,mode="RGB")
        im.save(image_name)




class Renderer_Composite(object):
    def __init__(self,calc,h_obs,lat_obs,lon_obs,max_distance,
                 distance_res=10,vert_obs_angle=0.0,vert_res=1000,
                 focal_length=2000,atol=1.1e-3,rtol=1.1e-5):


        if isinstance(calc,CurveCalc):
            self._sphere = True
            self._R0 = calc.R0
        elif isinstance(calc,FlatCalc):
            self._sphere = False
        else:
            raise ValueError

        self._calc = calc
        self._atol=atol
        self._rtol=rtol

        self._lat_obs = float(lat_obs)
        self._lon_obs = float(lon_obs)
        self._h_obs = float(h_obs)
        _check_gps(self._lat_obs,self._lon_obs)

        self._calc = calc
        self._geod = Geod(ellps="sphere")

        vert_res = int(vert_res)
        vert_min = np.rad2deg(np.arctan(-12.0/focal_length))+vert_obs_angle
        vert_max = np.rad2deg(np.arctan( 12.0/focal_length))+vert_obs_angle

        self._v_angles,self._dangles = np.linspace(vert_min,vert_max,vert_res,retstep=True)
        self._vfov = self._v_angles.max() - self._v_angles.min()

        self._ds = np.arange(0.0,max_distance,distance_res,dtype=np.float64)
        if self._sphere:
            sigmas = self._ds/self._R0
            sol = self._calc.solve_ivp(max_distance,h_obs,alpha=self._v_angles,atol=atol,rtol=rtol,dense_output=True)
            self._rs = (sol.sol(sigmas)[:vert_res]-self._R0)
        else:
            sol = self._calc.solve_ivp(max_distance,h_obs,alpha=self._v_angles,atol=atol,rtol=rtol,dense_output=True)
            self._rs = sol.sol(self._ds)[:vert_res].copy()

        
        self._sol = interp.interp1d(self._ds,self._rs,axis=-1,copy=False,bounds_error=True)

    @property
    def vfov(self):
        return self._vfov

    def render_scene(self,scene,image_names,heading_mins,heading_maxs,surface_color=None,background_color=None,cfunc=_defualt_cfunc,cfunc_args=(),disp=False,eye_level=False):
        if surface_color is None:
            surface_color = np.array([0,80,120])
        else:
            surface_color = np.fromiter(surface_color)

        if background_color is None:
            background_color = np.array([135,206,250])
        else:
            background_color = np.fromiter(background_color)

        img_datas = []
        ray_heights = {}

        for image,(im,image_data) in iteritems(scene._image_dict):
            im_data = np.array(im)
            im_data = im_data[::-1,:,:].transpose((1,0,2)).copy()

            for lat,lon,vert_pixel_pos,horz_pixel_pos,heading in image_data:
                v_px = vert_pixel_pos
                f_az,b_az,dist = self._geod.inv(self._lon_obs,self._lat_obs,lon,lat)

                if heading is None:
                    alpha = 0.0
                else:
                    b_az = b_az%360
                    heading = heading%360
                    alpha = np.deg2rad(heading - b_az)

                h_px=np.arctan(horz_pixel_pos*np.cos(alpha)/dist)
                np.rad2deg(h_px,out=h_px)
                h_px += (f_az % 360)
                np.mod(h_px,360,out=h_px)

                if dist not in ray_heights:
                    ray_heights[dist] = self._sol(dist)

                img_datas.append((im_data,h_px,v_px,dist))

        img_datas.sort(key=lambda x:-x[-1])
        
        n_v = self._rs.shape[0]

        land_model = scene._land_model

        for heading_min,heading_max,image_name in zip(heading_mins,heading_maxs,image_names):
            h_angles = np.arange(heading_min,heading_max,self._dangles)

            png_data = np.empty((len(h_angles),n_v,3),dtype=np.uint8)
            png_data[...] = 0

            terrain_args = (self._lat_obs,self._lon_obs,land_model,cfunc,cfunc_args)
            image_args = (img_datas,ray_heights)

            _render(png_data,self._rs,self._ds,h_angles,surface_color,background_color,terrain_args,image_args,disp)

            if eye_level:
                i_horz = np.argwhere(np.abs(self._v_angles)<(self._vfov/400.0)).ravel()
                png_data[:,i_horz,0] = 255
                png_data[:,i_horz,1] = 100
                png_data[:,i_horz,2] = 0

            png_data = png_data.transpose((1,0,2))
            png_data = png_data[::-1,:,:]
            im = Image.fromarray(png_data,mode="RGB")
            im.save(image_name)


class Scene(object):
    """
    Simple wrapper which keeps track of data used to render an image.
    """
    def __init__(self):
        """
        This function initialized `Scene` object.
        """
        self._land_model = land_model()
        self._image_dict = {}

    def add_elevation_model(self,lats,lons,elevation):
        """Add elevation data to the scene.

        Parameters
        ----------
        lats : array_like of shape (n,)
            list of latitudes which define the grid of data.

        lons : array_like of shape (m,)
            list of longitudes which define the grid of data.

        elevation : array_like of shape (n,m)
            list of elevation data at the points defined by the grid of `lats` and `lons`.
        """
        self._land_model.add_elevation_data(lats,lons,elevation)

    def add_image(self,image,image_pos,dimensions,direction=None):
        """Add image to scene. 

        Parameters
        ----------
        image: str
            string which contains path to image file. 

        image_pos: array_like
            either has (h0,lat,lon) or (lat,lon) h0 is height above the earth's surface

        dimensions: tuple
            contains dimensions of the image is in meters. If either one has value `-1` 
            then that dimension is determined by the resolution of the picture.
        """
        if type(image) is str:
            if image in self._image_dict:
                im = self._image_dict[image][0]
            else:
                im = Image.open(image)
                self._image_dict[image] = (im,[])
        else:
            raise ValueError("image must be a string containing name of file.")


        px_width,px_height = im.size

        aspect = float(px_height)/px_width

        if len(image_pos) == 2:
            lat,lon = image_pos
            h = self._land_model(lat,lon)
        elif len(image_pos) == 3:
            h,lat,lon = image_pos
        else:
            raise ValueError("expecting image_pos to contain gps coordinates and optionally height of image.")

        if direction is None:
            heading = None
        elif type(direction) is tuple:
            if len(direction) != 2:
                raise ValueError("direction must be either heading or tuple containing latitude and lonitude respectively.")

            lat_dir,lon_dir = direction
            _check_gps(lat_dir,lon_dir)
            heading,b_az,dist = self._geod.inv(lon_obs,lat_obs,lon_dir,lat_dir)

        else:
            heading = float(direction)

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


        self._image_dict[image][1].append((lat,lon,vert_pixel_pos,horz_pixel_pos,heading))


class land_model(object):
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
        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)
        lat,lon = np.broadcast_arrays(lat,lon)
        heights = np.zeros_like(lon)

        coors = np.stack((lat,lon),axis=-1)
        for terrain in self._terrain_list:
            heights += terrain(coors)

        return np.squeeze(heights)

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




