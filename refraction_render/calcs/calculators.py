from .standard_atmosphere import std_atmosphere
from .euler_equations import UniformFermatEquationsCurve,UniformFermatEquationsEuclid,FermatEquations
from scipy.integrate import solve_bvp,solve_ivp
import numpy as np

__all__=["Calc","CurveCalc","FlatCalc","CurveNoRefraction","FlatNoRefraction"]


class Calc(object):
    """Base class for Calculators."""
    def __init__(self,FermatEquations_obj):
        if not isinstance(FermatEquations_obj,FermatEquations):
            raise ValueError("FermatEquations_obj must be an instance of FermatEquations")
        self.FermatEquations_obj = FermatEquations_obj

    def solve_ivp(self,d,h,dh=None,alpha=None,**kwargs):
        """Solve initial value problem for light rays.

        d: float
            maximum distance to solve the light rays.
        h: array_like (n,)
            initial height of rays.
        dh: array_like (n,), optional
            initial derivatives of the rays.
        alpha:  array_like (n,), optional
            initial angle of the rays.
        **kwargs: optional
            extra arguments which get passed into Fermat equatino solver.

        """
        if alpha is not None and dh is None:
            h,dh = np.broadcast_arrays(h,np.tan(np.deg2rad(alpha)))
        elif alpha is None and dh is not None:
            h,dh = np.broadcast_arrays(h,dh)
        elif alpha is not None and dh is not None:
            raise ValueError("use only 'alpha' or 'dh' not both.")
        else:
            raise ValueError("in order to complete initial values you need 'alpha' or 'dh'.")

        return self.FermatEquations_obj.solve_ivp(0,d,h,dh,**kwargs)


class CurveCalc(Calc):
    """Calculator used for calculating rays on a spherical earth in an atmosphere."""
    def __init__(self,R0=6370997.0,**std_atmosphere_args):
        """Iintialize 'CurveCalc' object.

        R0: float, optional
            Radius of the sphere for this calculation.
        **std_atmosphere_args: optional
            arguments to 'std_atmosphere' object. 
        """
        self._R0 = R0
        self._atm = std_atmosphere(**std_atmosphere_args)
        Calc.__init__(self,UniformFermatEquationsCurve(R0,self._atm._n,self._atm._dndy))

    @property
    def atm_model(self):
        """Atmospheric model used for this calculator."""
        return self._atm

    @property
    def R0(self):
        """Radius of the sphere in this calculator."""
        return self._R0


class FlatCalc(Calc):
    """Calculator used for calculating rays on a flat earth in an atmosphere."""
    def __init__(self,**std_atmosphere_args):
        """Initialize 'FlatCalc' object.

        **std_atmosphere_args: optional
            arguments to 'std_atmosphere' object. 
        """
        self._atm = std_atmosphere(**std_atmosphere_args)
        Calc.__init__(self,UniformFermatEquationsEuclid(self._atm._n,self._atm._dndy))

    @property
    def atm_model(self):
        """Atmospheric model used for this calculator."""
        return self._atm


class CurveNoRefraction(Calc):
    """Calculator used for calculating rays on a sphere earth in no atmosphere."""
    def __init__(self,R0=6370997.0):
        """Initialize 'CurveNoRefraction' object.

        R0: float, optional
            Radius of the sphere for this calculation.
        """
        self._R0 = R0
        Calc.__init__(self,UniformFermatEquationsCurve(R0,lambda s,y:1.0,lambda s,y:0.0))

    @property
    def R0(self):
        """Radius of the sphere in this calculator."""
        return self._R0


class FlatNoRefraction(Calc):
    """Calculator used for calculating rays on a flat earth in no atmosphere."""
    def __init__(self):
        """Initialize 'FlatNoRefraction' object."""
        Calc.__init__(self,UniformFermatEquationsEuclid(lambda s,y:1.0,lambda s,y:0.0))


















