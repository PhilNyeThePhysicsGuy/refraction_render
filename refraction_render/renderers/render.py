from __future__ import division,print_function

from ..calcs import Calc

from numba import njit,guvectorize,vectorize,prange
import numba.cuda as cuda
from PIL import Image
from pyproj import Geod
from six import iteritems
import scipy.interpolate as interp
import scipy.signal as signal
import numpy as np
from tqdm import tqdm

__all__=["Scene","Renderer_35mm","Renderer_Composite","land_model","ray_diagram"]


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
def _get_bounds_block(a,a_mins,a_maxs,mask):
    n = a.shape[0]
    m = a_mins.shape[0]
    for i in range(m):
        mask[i] = False
        for j in range(n):
            if a[j] >= a_mins[i] and a[j] < a_maxs[i]:
                mask[i] = True
                break

@njit
def _get_bounds_sum(a,a_min,a_max):
    n = a.shape[0]
    count = 0
    for i in range(n):
        if a[i] >= a_min and a[i] < a_max:
            count += 1

    return count

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


@cuda.jit("void(f8,f8[:,::1],f8[:],i4[:],b1[:],b1[:],b1[:])")
def _ray_crossing_gpu(h_min,rs,heights,inds,water,land,sky):
    i = cuda.grid(1)
    n_d = rs.shape[1]

    if i < rs.shape[0]:
        hit = False
        water[i] = False
        land[i] = False
        sky[i] = False
        inds[i] = -1
        for j in range(n_d):
            if rs[i,j] <= heights[j]:
                hit = True
                if heights[j] > h_min:
                    land[i] = True
                    inds[i] = j
                else:
                    water[i] = True
                    inds[i] = j

                break

        sky[i] = not hit


@njit(["void(f8,f8[:,::1],f8[:],i4[:],b1[:],b1[:],b1[:])"])
def _ray_crossing_cpu(h_min,rs,heights,inds,water,land,sky):
    n_v = rs.shape[0]
    n_d = rs.shape[1]

    water[:] = False
    land[:] = False
    sky[:] = False
    inds[:] = -1
    
    for i in range(n_v):
        hit = False

        for j in range(n_d):
            if rs[i,j] <= heights[j]:
                hit = True
                if heights[j] > h_min:
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
            alpha = img_png_data[kk,l[b],3] / (255.0)
            png_data[ii,j[b],:] = (1.0 - alpha)*png_data[ii,j[b],:] + alpha*img_png_data[kk,l[b],:3]

@njit
def _update_png_data_slice(j,png_data,l,img_png_data):
    m = j.shape[0]

    for b in range(m):
       if img_png_data[l[b],3] > 0:
            alpha = img_png_data[l[b],3] / (255.0)
            png_data[j[b],:] =  (1.0 - alpha)*png_data[j[b],:] + alpha*img_png_data[l[b],:3]

def _check_gps(lat,lon):
    if np.any(lat < -90) or np.any(lat > 90):
        raise ValueError("latitude must be between -90 and 90")
    if np.any(lon < -180) or np.any(lon > 180):
        raise ValueError("lonitude must be between -180 and 180")

def _defualt_cfunc(d,heights,background_color):
    alpha = np.asarray(np.exp(-d/200000.0)) # exponential attenuation of light due to atmospheric scattering
    out = np.zeros(alpha.shape+(3,),dtype=np.uint8)

    out[:,0] = 50*alpha + background_color[0]*(1-alpha)
    out[:,1] = 150*alpha + background_color[1]*(1-alpha)
    out[:,2] = background_color[2]*(1-alpha)

    return out


def _render_cpu(png_data,h_min,rs,ds,h_angles,surface_color,background_color,terrain_args,image_args,disp=False):

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

    h_mins = np.array([h_px.min() for _,h_px,_,_ in img_datas])
    h_maxs = np.array([h_px.max() for _,h_px,_,_ in img_datas])
    img_mask = np.zeros_like(h_mins,dtype=np.bool)

    if terrain.has_data: # render land model 


        h_angle_max = h_angles.max()
        if disp:
            h_iter = tqdm(h_angles)
        else:
            h_iter = iter(h_angles)

        for i,h_angle in enumerate(h_iter):

            heights = terrain.get_terrain(lat_obs,lon_obs,h_angle,ds)

            _ray_crossing_cpu(h_min,rs,heights,inds,water,land,sky)

            png_data[i,water,:] = surface_color
            png_data[i,sky,:] = background_color

            if np.any(land):
                iland, = np.where(land)
                land_inds = inds[land]
                land_inds_m = land_inds-1
                land_inds_m[land_inds_m<0] = 0

                _heights = heights[land_inds]
                dx = ds[land_inds] - ds[land_inds_m]
                dy = (rs[iland,land_inds] - rs[iland,land_inds_m])
                dh = (_heights - heights[land_inds_m])

                t_ray = np.stack((dx,dy),axis=-1)
                t_land = np.stack((dx,dh),axis=-1)
                t_ray = (t_ray.T/np.linalg.norm(t_ray,axis=1))
                t_land = (t_land.T/np.linalg.norm(t_land,axis=1))
                n_ray = t_ray - 2*np.sum(t_ray*t_land,axis=0)*t_land

                try:
                    png_data[i,land,:] = cfunc(ds[land_inds],_heights,n_ray,*cfunc_args)
                except TypeError:
                    png_data[i,land,:] = cfunc(ds[land_inds],_heights,*cfunc_args)

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

        _get_bounds_block(h_angles,h_mins,h_maxs,img_mask)
        if np.any(img_mask):
            img_indx = np.argwhere(img_mask).ravel()
            for I in img_indx:
                img_png_data,h_px,v_px,d = img_datas[I]
                
                rh = ray_heights[d]
                _get_bounds(h_angles,h_px[0],h_px[-1],out=h_mask)
                _get_vertical_mask(rh,v_px[0],v_px[-1],inds,ds,d,sky,v_mask)
                if np.any(v_mask):
                    i = np.argwhere(h_mask).ravel()
                    j = np.argwhere(v_mask).ravel()
                    k = np.searchsorted(h_px,h_angles[i])
                    l = np.searchsorted(v_px,rh[j])

                    _update_png_data(i,j,png_data,k,l,img_png_data)


def _render_gpu(png_data,h_min,rs,ds,h_angles,surface_color,background_color,terrain_args,image_args,disp=False):

    lat_obs,lon_obs,terrain,cfunc,cfunc_args = terrain_args
    img_datas,ray_heights = image_args

    if terrain.has_data: # render land model 
        n_v = rs.shape[0]
        n_h = h_angles.shape[0]
        n_z = rs.shape[1]

        water = np.zeros(n_v,dtype=np.bool)
        land = np.zeros(n_v,dtype=np.bool)
        sky = np.zeros(n_v,dtype=np.bool)
        inds = np.zeros(n_v,dtype=np.int32)

        water_dev = cuda.to_device(water)
        land_dev = cuda.to_device(land)
        sky_dev = cuda.to_device(sky)
        inds_dev = cuda.to_device(inds)
        heights_dev = cuda.device_array_like(ds)
        ds_dev = cuda.to_device(ds)
        rs_dev = cuda.to_device(rs)

        stream = cuda.stream()

        v_mask = np.zeros(n_v,dtype=np.bool)
        h_mask = np.zeros(h_angles.shape[0],dtype=np.bool)

        h_mins = np.array([h_px.min() for _,h_px,_,_ in img_datas])
        h_maxs = np.array([h_px.max() for _,h_px,_,_ in img_datas])
        img_mask = np.zeros_like(h_mins,dtype=np.bool)

        nth = 1024
        nbk = max(n_v//nth,1)+1
        print(n_v,nth,nbk)

        h_angle_max = h_angles.max()

        if disp:
            h_iter = tqdm(h_angles)
        else:
            h_iter = iter(h_angles)

        for i,h_angle in enumerate(h_iter):
            heights = terrain.get_terrain(lat_obs,lon_obs,h_angle,ds)

            cuda.to_device(heights,to=heights_dev)

            _ray_crossing_gpu[nbk,nth](h_min,rs_dev,heights_dev,inds_dev,water_dev,land_dev,sky_dev)

            water_dev.copy_to_host(ary=water, stream=stream)
            inds_dev.copy_to_host(ary=inds, stream=stream)
            land_dev.copy_to_host(ary=land, stream=stream)
            sky_dev.copy_to_host(ary=sky, stream=stream)

            png_data[i,water,:] = surface_color
            png_data[i,sky,:] = background_color

            if np.any(land):
                iland, = np.where(land)
                land_inds = inds[land]
                land_inds_m = land_inds-1
                land_inds_m[land_inds_m<0] = 0

                _heights = heights[land_inds]
                dx = ds[land_inds] - ds[land_inds_m]
                dy = (rs[iland,land_inds] - rs[iland,land_inds_m])
                dh = (_heights - heights[land_inds_m])

                t_ray = np.stack((dx,dy),axis=-1)
                t_land = np.stack((dx,dh),axis=-1)
                t_ray = (t_ray.T/np.linalg.norm(t_ray,axis=1))
                t_land = (t_land.T/np.linalg.norm(t_land,axis=1))
                n_ray = t_ray - 2*np.sum(t_ray*t_land,axis=0)*t_land

                try:
                    png_data[i,land,:] = cfunc(ds[land_inds],_heights,n_ray,*cfunc_args)
                except TypeError:
                    png_data[i,land,:] = cfunc(ds[land_inds],_heights,*cfunc_args)

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

        n_v = rs.shape[0]
        n_z = rs.shape[1]

        water = np.zeros(n_v,dtype=np.bool)
        sky = np.zeros(n_v,dtype=np.bool)
        inds = np.zeros(n_v,dtype=np.int32)

        v_mask = np.zeros(n_v,dtype=np.bool)
        h_mask = np.zeros(h_angles.shape[0],dtype=np.bool)

        h_mins = np.array([h_px.min() for _,h_px,_,_ in img_datas])
        h_maxs = np.array([h_px.max() for _,h_px,_,_ in img_datas])
        img_mask = np.zeros_like(h_mins,dtype=np.bool)

        nth = 1024
        nbk = max(n_v//nth,1)

        _get_water(rs,inds)
        np.less(inds,n_z,out=water)
        png_data[:,water,:] = surface_color
        np.logical_not(water,out=sky)
        png_data[:,sky,:] = background_color

        _get_bounds_block(h_angles,h_mins,h_maxs,img_mask)
        if np.any(img_mask):
            img_indx = np.argwhere(img_mask).ravel()
            for I in img_indx:
                img_png_data,h_px,v_px,d = img_datas[I]
                
                rh = ray_heights[d]
                _get_bounds(h_angles,h_px[0],h_px[-1],out=h_mask)
                _get_vertical_mask(rh,v_px[0],v_px[-1],inds,ds,d,sky,v_mask)
                if np.any(v_mask):
                    i = np.argwhere(h_mask).ravel()
                    j = np.argwhere(v_mask).ravel()
                    k = np.searchsorted(h_px,h_angles[i])
                    l = np.searchsorted(v_px,rh[j])

                    _update_png_data(i,j,png_data,k,l,img_png_data)


def _prep_scene(scene,h_angles,lat_obs,lon_obs,geod,sol):
    img_datas = []
    ray_heights = {}
    dh_angle = np.diff(h_angles).min()

    for image,(im,image_data) in iteritems(scene._image_dict):
        px_width,px_height = im.size

        for lat,lon,img_height,width,height,heading in image_data:
            horz_pixel_pos = np.linspace(-width/2,width/2,px_width)
            v_px = img_height + np.linspace(0,height,px_height)

            f_az,b_az,dist = geod.inv(lon_obs,lat_obs,lon,lat)

            if heading is None:
                alpha = 0.0
            else:
                b_az = b_az%360
                heading = heading%360
                alpha = np.deg2rad(heading - b_az)

            h_px=np.arctan(horz_pixel_pos*np.abs(np.cos(alpha))/dist)

            np.rad2deg(h_px,out=h_px)
            h_px += (f_az % 360)
            np.mod(h_px,360,out=h_px)

            if dist not in ray_heights:
                rh = sol(dist)
                ray_heights[dist] = rh
            else:
                rh = ray_heights[dist]

            n_h = max(int((h_px[-1]-h_px[0])/dh_angle),2)
            n_v = max(_get_bounds_sum(rh,v_px[0],v_px[-1]),2)

            try:
                new_im = im.resize((n_h,n_v),Image.LANCZOS)
            except ValueError:
                continue
            im_data = np.array(new_im)
            im_data = im_data[::-1,:,:].transpose((1,0,2)).copy()
            h_px = np.linspace(h_px[0],h_px[-1],n_h)
            v_px = np.linspace(v_px[0],v_px[-1],n_v)
            img_datas.append((im_data,h_px,v_px,dist))


    img_datas.sort(key=lambda x:-x[-1])

    return img_datas,ray_heights

@njit
def is_sorted(a):
    for i in range(a.size-1):
         if a[i+1] < a[i] :
               return False
    return True

def ray_diagram(ax,calc,h_obs,d,angles,heights=None,style="sphere_top",
                eye_level=True,linewidth_rays=0.2,linewidth_earth=0.2,R0=6371008,h_min=0.01):

    """Side profile view showing the rays trajectories. 

    Parameters
    ----------
    ax: matplotlib Axes object
        this function plots lines on a graph, you can specify which graph to plot to by passing in the axes object for the graph you want to plot to.


    calc: calcs object
        object which can be used to calculate the trajectory of light rays

    h_obs: float
        height of renderer in meters
    
    d: array_like, (N,)
        list of values to evaluate ray positions at for ray diagram, must be sorted in ascending order.

    angles: array_like, (M,)
        list of initial angles for the rays, in degrees

    heights: array_like, (N,), optional
        list of values which represent the elevation profile along ray trajectory

    style: str, optional 
        style to plot the graph. "flat": plot rays on flat plane, "sphere_top": plot rays as if the earth is filling away from the observer, "sphere_side": plot rays with the 'bulge' in the middle. 

    eye_level: bool, optional
        color the ray which is closest to eye level (0 degrees initial angle) orange.

    linewidth_rays: float, optional
        linewidth used for plotting rays in diagram

    linewidth_earth: float, optional
        linewidth used for plotting surface of the earth, both water and land.

    h_min: float, optional
        minimum values which should count as water when calculating hit locations.

    """
    
    if len(d) == 0 or not is_sorted(d):
        raise ValueError("array 'd' must contain distance values in ascending order.")


    angles = np.asarray(angles).ravel()
    d_max = d.max()

    sol = calc.solve_ivp(d.max(),h_obs,alpha=angles,dense_output=True,atol=1.1e-10,rtol=1.1e-7)
    n_v = angles.shape[0]

    rs = sol.sol(d)[:n_v].copy()

    water = np.zeros(n_v,dtype=np.bool)
    land = np.zeros(n_v,dtype=np.bool)
    sky = np.zeros(n_v,dtype=np.bool)
    inds = np.zeros(n_v,dtype=np.int32)

    if angles.size > 0:
        i_horz = np.abs(angles).argmin()

    if heights is None:
        heights = np.zeros_like(d)
    else:
        heights = heights.astype(np.float64,copy=False).ravel()
        if len(heights) != len(d):
            raise ValueError("number of elevation points must match the number of positions.")
        

    if style == "sphere_top":
        c = np.cos(np.pi/2-d/R0)
        s = np.sin(np.pi/2-d/R0)

        _ray_crossing_cpu(h_min,rs,heights,inds,water,land,sky)
        rs += R0
        heights = heights + R0
        for i in range(n_v):
            if i == i_horz and eye_level:
                i_max = inds[i]
                ax.plot(rs[i,:i_max]*c[:i_max],
                        rs[i,:i_max]*s[:i_max]-R0,color="orange",linewidth=linewidth_rays)
                continue

            if water[i]:
                i_max = inds[i]
                ax.plot(rs[i,:i_max]*c[:i_max],
                        rs[i,:i_max]*s[:i_max]-R0,color="blue",linewidth=linewidth_rays)
                continue

            if land[i]:
                i_max = inds[i]
                ax.plot(rs[i,:i_max]*c[:i_max],
                        rs[i,:i_max]*s[:i_max]-R0,color="green",linewidth=linewidth_rays)
                continue

            if sky[i]:
                ax.plot(rs[i,:]*c,rs[i,:]*s-R0,color="cyan",linewidth=linewidth_rays)
                continue

        ax.plot(c*heights,s*heights-R0,color="green",linewidth=linewidth_earth)
        ax.plot(c*R0,s*R0-R0,color="blue",linewidth=linewidth_earth)
    elif style == "sphere_side":
        c = np.cos(np.pi/2-d/R0+d_max/(2*R0))
        c -= c.min()
        s = np.sin(np.pi/2-d/R0+d_max/(2*R0))
        _ray_crossing_cpu(h_min,rs,heights,inds,water,land,sky)
        rs += R0
        heights = heights + R0
        for i in range(n_v):
            if i == i_horz and eye_level:
                i_max = inds[i]
                ax.plot(rs[i,:i_max]*c[:i_max],
                        rs[i,:i_max]*s[:i_max]-R0*s.min(),color="orange",linewidth=linewidth_rays)
                continue

            if water[i]:
                i_max = inds[i]
                ax.plot(rs[i,:i_max]*c[:i_max],
                        rs[i,:i_max]*s[:i_max]-R0*s.min(),color="blue",linewidth=linewidth_rays)
                continue

            if land[i]:
                i_max = inds[i]
                ax.plot(rs[i,:i_max]*c[:i_max],
                        rs[i,:i_max]*s[:i_max]-R0*s.min(),color="green",linewidth=linewidth_rays)
                continue

            if sky[i]:
                ax.plot(rs[i,:]*c,rs[i,:]*s-R0*s.min(),color="cyan",linewidth=linewidth_rays)
                continue

        ax.plot(c*heights,s*heights-R0*s.min(),color="green",linewidth=linewidth_earth)
        ax.plot(c*R0,s*R0-R0*s.min(),color="blue",linewidth=linewidth_earth)
    elif style == "flat":
        _ray_crossing_cpu(h_min,rs,heights,inds,water,land,sky)

        for i in range(n_v):
            if i == i_horz and eye_level:
                i_max = inds[i]
                ax.plot(d[:i_max],rs[i,:i_max],color="orange",linewidth=linewidth_rays)
                continue

            if water[i]:
                i_max = inds[i]
                ax.plot(d[:i_max],rs[i,:i_max],color="blue",linewidth=linewidth_rays)
                continue

            if land[i]:
                i_max = inds[i]
                ax.plot(d[:i_max],rs[i,:i_max],color="green",linewidth=linewidth_rays)
                continue

            if sky[i]:
                ax.plot(d,rs[i,:],color="cyan",linewidth=linewidth_rays)
                continue

        ax.plot(d,heights,color="green",linewidth=linewidth_earth)
        ax.plot(d,np.zeros_like(d),color="blue",linewidth=linewidth_earth)
    else:
        raise ValueError


class Renderer_35mm(object):
    """
    Object used to set a camera angle and position to render a scene.
    """
    def __init__(self,calc,h_obs,lat_obs,lon_obs,direction,max_distance,
                 distance_res=10,vert_obs_angle=0.0,vert_res=1000,
                 focal_length=2000,atol=1.1e-7,rtol=1.1e-7,aspect_ratio=None):

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
            focal length in milimeters to calculate the field of view of the camera. This, along with the vertial resolution, sets the resolution on the horizontal.

        atol: float
            absolute tolerance of ode solver

        rtol: float
            relative tolerance of the ode solver

        aspect_ratio: tuple
            aspect ratio for image

        Note
        ----
        The focal length sets the verticle field of view of the frame using the standard 24 mm for 35 mm film. 

        """
        if not isinstance(calc,Calc):
            raise ValueError("expecting calculator to be instance of Calc base.")


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

        if aspect_ratio is None:
            dx=18
            horz_res = int(vert_res*1.5)
        else:
            dx = float(aspect_ratio[0])*12.0/float(aspect_ratio[1])
            r = float(aspect_ratio[0])/aspect_ratio[1]
            horz_res = int(vert_res*r)



        self._vert_res = vert_res
        self._horz_res = horz_res
        self._focal_length = focal_length

        y_grid = np.linspace(-12,12,vert_res)
        x_grid = np.linspace(-dx,dx,horz_res)
        self._v_angles = np.rad2deg(np.arctan(y_grid/focal_length))+vert_obs_angle
        self._h_angles = np.rad2deg(np.arctan(x_grid/focal_length))+f_az
        self._vfov = self._v_angles.max()-self._v_angles.min()

        self._ds = np.arange(0.0,max_distance,distance_res,dtype=np.float64)
        sol = self._calc.solve_ivp(max_distance,h_obs,alpha=self._v_angles,atol=atol,rtol=rtol,dense_output=True)
        self._rs = np.ascontiguousarray(sol.sol(self._ds)[:vert_res])

        
        self._sol = interp.interp1d(self._ds,self._rs,axis=-1,copy=False,bounds_error=True)

    @property
    def vfov(self):
        """vertial field of view for this particular renderer"""
        return self._vfov

    @property
    def v_angles(self):
        """vertical angular scale of the image frame"""
        v_angles = self._v_angles[...]
        v_angles.setflags(write=False)
        return v_angles

    @property
    def h_angles(self):
        """horizontal angular scale of the image frame"""
        h_angles = self._h_angles[...]
        h_angles.setflags(write=False)
        return h_angles

    @property
    def calc(self):
        return self._calc

    def set_location(self,lat_obs,lon_obs,direction):
        """This function can be used to change the location and heading of the renderer.

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


    def render_scene(self,scene,image_name,surface_color=None,background_color=None,cfunc=None,cfunc_args=None,
        disp=False,eye_level=False,postprocess=None,h_min=0.01,gpu=False):
        """Render a scene object for the renderer's given field of view and direction. 

        Parameters
        ----------
        scene: Scene object
            object which contains data which the renderer can extract and render

        image_name: str
            name for the image

        background_color: array_like, optional
            3 element array containing an color via RGB color code (numbered 0 to 255)
            default value: `[135,206,250]`

        surface_color: array_like, optional
            3 element array containing an color via RGB color code (numbered 0 to 255)
            default value: `[0,80,120]`

        cfunc: callable, optional
            Callable function which is used to color the elevation data. The function takes 
            in two arrays, the first is distances from the observer    the second is the elevation
            above the surface of the earth. 

        cfunc_args: array_like, optional
            extra arguments to pass into `cfunc`

        disp: bool, optional
            when rendering topographical data this will print out the heading slice which has been rendered.

        eye_level: bool, optional
            when rendering the image, an orange line is placed at eye level in the image.

        post_process: callable, optional
            function which processes the final image before saving it.

        h_min: float, optional
            minimum value for ray to count as crossing water. 

        gpu: bool, optional
            utilize GPU to calculate 3D land model rendering.

        """        
        if surface_color is None:
            surface_color = np.array([0,80,120],dtype=np.uint8)
        else:
            surface_color = np.fromiter(surface_color,dtype=np.uint8)

        if background_color is None:
            background_color = np.array([135,206,250],dtype=np.uint8)
        else:
            background_color = np.fromiter(background_color,dtype=np.uint8)

        if cfunc is None and cfunc_args is None:
            cfunc = _defualt_cfunc
            cfunc_args = (background_color,)
        elif cfunc is None and cfunc_args is not None:
            raise Exception("cfunc_args given without cfunc.")

        if cfunc_args is None:
            cfunc_args = ()

        img_datas,ray_heights = _prep_scene(scene,self._h_angles,self._lat_obs,self._lon_obs,self._geod,self._sol)

        land_model = scene._land_model

        png_data = np.empty((len(self._h_angles),self._rs.shape[0],3),dtype=np.uint8)
        png_data[...] = 0

        terrain_args = (self._lat_obs,self._lon_obs,land_model,cfunc,cfunc_args)
        image_args = (img_datas,ray_heights)


        if gpu:
            _render_gpu(png_data,h_min,self._rs,self._ds,self._h_angles,surface_color,background_color,terrain_args,image_args,disp)
        else:
            _render_cpu(png_data,h_min,self._rs,self._ds,self._h_angles,surface_color,background_color,terrain_args,image_args,disp)

        if eye_level:
            i_horz = np.argwhere(np.abs(self._v_angles)<(self._vfov/800.0)).ravel()
            png_data[:,i_horz,0] = 255
            png_data[:,i_horz,1] = 100
            png_data[:,i_horz,2] = 0


        png_data = png_data.transpose((1,0,2))
        png_data = png_data[::-1,:,:]
        im = Image.fromarray(png_data,mode="RGB")

        if postprocess is not None:
            im = postprocess(im)

        im.save(image_name)


class Renderer_Composite(object):
    """
    Object used to set a camera angle and position to render a scene for a user specified horizontal field of view. 
    """
    def __init__(self,calc,h_obs,lat_obs,lon_obs,max_distance,
                 distance_res=10,vert_obs_angle=0.0,vert_res=1000,
                 focal_length=2000,atol=1.1e-7,rtol=1.1e-7):
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

        max_distance: float
            maximum distance away from the renderer to calculate the light ray path

        distance_res: float
            distance between points where the light ray path is checked for intersection point.

        vert_obs_angle: float
            vertical tilt of the renderer. 

        vert_res: int
            resolution of pixels on the verticle

        focal_length: float
            focal length in milimeters to calculate the field of view of the camera. This, along with the vertial resolution, sets the resolution on the horizontal.

        atol: float
            absolute tolerance of ode solver

        rtol: float
            relative tolerance of the ode solver


        """

        if not isinstance(calc,Calc):
            raise ValueError("expecting calculator to be instance of Calc base.")

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
        sol = self._calc.solve_ivp(max_distance,h_obs,alpha=self._v_angles,atol=atol,rtol=rtol,dense_output=True)
        self._rs = sol.sol(self._ds)[:vert_res].copy()

        self._sol = interp.interp1d(self._ds,self._rs,axis=-1,copy=False,bounds_error=True)

    @property
    def vfov(self):
        """vertial field of view for this particular renderer"""
        return self._vfov

    @property
    def v_angles(self):
        """vertical angular scale of the image frame"""
        v_angles = self._v_angles[...]
        v_angles.setflags(write=False)
        return v_angles

    def set_location(self,lat_obs,lon_obs):
        """This function can be used to change the position of the renderer.

        Parameters
        ----------
        lat_obs: float
            new latitude of observer
        lon_obs: float
            new longitude of observer

        """
        self._lat_obs = float(lat_obs)
        self._lon_obs = float(lon_obs)
        _check_gps(self._lat_obs,self._lon_obs)

    def render_scene(self,scene,image_names,heading_mins,heading_maxs,surface_color=None,background_color=None,
        cfunc=None,cfunc_args=None,disp=False,eye_level=False,postprocess=None,h_min=0.01,gpu=False):
        """Renders a composites over a very wide horizontal field.

        Parameters
        ----------
        scene: Scene object
            object which contains data which the renderer can extract and render

        image_names: array_like (n,) or 'str' 
            name(s) for the image(s) being rendered.

        heading_mins: array_like (n,) or float
            minimum starting point(s) for composite.

        heading_maxs: array_like (n,) or float
            maximum starting point(s) for composite, pair with 'heading_mins'.

        background_color: array_like, optional
            3 element array containing an color via RGB color code (numbered 0 to 255)
            default value: `[135,206,250]`

        surface_color: array_like, optional
            3 element array containing an color via RGB color code (numbered 0 to 255)
            default value: `[0,80,120]`

        cfunc: callable, optional
            Callable function which is used to color the elevation data. The function takes 
            in two arrays, the first is distances from the observer    the second is the elevation
            above the surface of the earth. 

        cfunc_args: array_like, optional
            extra arguments to pass into `cfunc`

        disp: bool, optional
            when rendering topographical data this will print out the heading slice which has been rendered.

        eye_level: bool, optional
            when rendering the image, an orange line is placed at eye level in the image.

        post_process: callable, optional
            function which processes the final image before saving it.

        h_min: float, optional
            minimum value for ray to count as crossing water. 

        gpu: bool, optional
            utilize GPU to calculate 3D land model rendering.

        """  
        if surface_color is None:
            surface_color = np.array([0,80,120])
        else:
            surface_color = np.fromiter(surface_color,dtype=np.uint8)

        if background_color is None:
            background_color = np.array([135,206,250])
        else:
            background_color = np.fromiter(background_color,dtype=np.uint8)

        if cfunc is None and cfunc_args is None:
            cfunc = _defualt_cfunc
            cfunc_args = (background_color,)
        elif cfunc is None and cfunc_args is not None:
            raise Exception("cfunc_args given without cfunc.")

        if cfunc_args is None:
            cfunc_args = ()

        n_v = self._rs.shape[0]
        
        
        land_model = scene._land_model

        heading_mins = np.atleast_1d(heading_mins).ravel()
        heading_maxs = np.atleast_1d(heading_maxs).ravel()
        image_names =  np.atleast_1d(image_names).ravel()
        tup = np.broadcast_arrays(heading_mins,heading_maxs,image_names)

        for heading_min,heading_max,image_name in zip(*tup):
            print(heading_min,heading_max,image_name)
            h_angles = np.arange(heading_min,heading_max,self._dangles)

            img_datas,ray_heights = _prep_scene(scene,h_angles,self._lat_obs,self._lon_obs,self._geod,self._sol)

            png_data = np.empty((len(h_angles),n_v,3),dtype=np.uint8)
            png_data[...] = 0

            terrain_args = (self._lat_obs,self._lon_obs,land_model,cfunc,cfunc_args)
            image_args = (img_datas,ray_heights)

            if gpu:
                _render_gpu(png_data,h_min,self._rs,self._ds,h_angles,surface_color,background_color,terrain_args,image_args,disp)
            else:
                _render_cpu(png_data,h_min,self._rs,self._ds,h_angles,surface_color,background_color,terrain_args,image_args,disp)

            if eye_level:
                i_horz = np.argwhere(np.abs(self._v_angles)<(self._vfov/800.0)).ravel()
                png_data[:,i_horz,0] = 255
                png_data[:,i_horz,1] = 100
                png_data[:,i_horz,2] = 0

            png_data = png_data.transpose((1,0,2))
            png_data = png_data[::-1,:,:]
            im = Image.fromarray(png_data,mode="RGB")
            if postprocess is not None:
                im = postprocess(im)
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

    @property
    def land_model(self):
        return self._land_model

    def add_elevation_model(self,*args):
        """Add terrain data to the interpolated model.

        Parameters
        ----------
        args: tuple
            tuple which contains elevation data:
            if len(args) == 3: args = (lats,lons,elevation) which contains the arguments for scipy.interpolate.RegularGridInterpolator.
            if len(args) == 2: args = (points,elevation) which contains the arguments for scipy.interpolate.LinearNDInterpolator.

        """
        if len(args)==1 and isinstance(args[0],land_model):
            self._land_model = args[0]
        else:
            self._land_model.add_elevation_data(*args)

    def add_image(self,image,image_pos,dimensions,direction=None):
        """Add image to scene. 

        Parameters
        ----------
        image: str
            string which contains path to image file. 

        image_pos: array_like
            either has (h_obj,lat,lon) or (lat,lon) h_obj is height above the earth's surface

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


        self._image_dict[image][1].append((lat,lon,h,width,height,heading))


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
        """flag if True the terrain object has land else it has no land."""
        return len(self._terrain_list) > 0

    def add_elevation_data(self,*args):
        """Add terrain data to the interpolated model.

        Parameters
        ----------
        args: tuple
            tuple which contains elevation data:
            if len(args) == 3: args = (lats,lons,elevation) which contains the arguments for scipy.interpolate.RegularGridInterpolator.
            if len(args) == 2: args = (points,elevation) which contains the arguments for scipy.interpolate.LinearNDInterpolator.

        """
        if len(args) == 3:
            lats,lons,elevation = args
            self._terrain_list.append(interp.RegularGridInterpolator((lats,lons),elevation,bounds_error=False,fill_value=0.0,method="linear"))
        elif len(args) == 2:
            points,elevation = args
            self._terrain_list.append(interp.LinearNDInterpolator(points,elevation,fill_value=0.0))
        else:
            raise ValueError("can't interpret arguments.")




