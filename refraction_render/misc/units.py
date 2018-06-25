import numpy as _np


def mi_to_m(d,out=None):
	return _np.multiply(1609.344,d,out=out)

def ft_to_m(d,out=None):
	return _np.multiply(0.3048,d,out=out)

def km_to_m(d,out=None):
	return _np.multiply(1000.0,d,out=out)