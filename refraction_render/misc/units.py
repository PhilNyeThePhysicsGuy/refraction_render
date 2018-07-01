import numpy as _np


def mi_to_m(d,out=None):
	"""Converts miles to meters.

	Examples
	--------
	>>> dist_meters = mi_to_m(dist_miles)

	Parameters
	----------
	d : array_like,scalar
		array / value in miles to convert to meters
	out : array_like, optional
		buffer for output values to go to. 

	"""
	return _np.multiply(1609.344,d,out=out)

def ft_to_m(d,out=None):
	"""Converts feet to meters.

	Examples
	--------
	>>> dist_meters = mi_to_m(dist_feet)

	Parameters
	----------
	d : array_like,scalar
		array / value in feet to convert to meters
	out : array_like, optional
		buffer for output values to go to. 

	"""
	return _np.multiply(0.3048,d,out=out)

def km_to_m(d,out=None):
	"""Converts kilometers to meters.

	Examples
	--------
	>>> dist_meters = mi_to_m(dist_kilometers)

	Parameters
	----------
	d : array_like,scalar
		array / value in milometers to convert to meters
	out : array_like, optional
		buffer for output values to go to. 
		
	"""
	return _np.multiply(1000.0,d,out=out)