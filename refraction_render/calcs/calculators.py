from .standard_atmosphere import std_atmosphere
from .euler_equations import UniformFermatEquationsCurve,UniformFermatEquationsEuclid,_EulerEquations
from scipy.integrate import solve_bvp,solve_ivp
import numpy as np

__all__=["CurveCalc","FlatCalc","Calc"]


class Calc(object):
    def __init__(self,EulerEquation_obj):
        self.EulerEquation_obj = EulerEquation_obj

    def solve_ivp(self,d,h,dh=None,alpha=None,**kwargs):
        if alpha is not None:
            dh = np.tan(np.deg2rad(alpha))
        
        h,dh = np.broadcast_arrays(h,dh)

        return self.EulerEquation_obj.solve_ivp(0,d,h,dh,**kwargs)


class CurveCalc(Calc):
    def __init__(self,R0=6370997.0,**std_atmosphere_args):
        self._R0 = R0
        self._atm = std_atmosphere(**std_atmosphere_args)
        Calc.__init__(self,UniformFermatEquationsCurve(R0,self._atm._n,self._atm._dndy))

    @property
    def atm_model(self):
        return self._atm

    @property
    def R0(self):
        return self._R0


class FlatCalc(Calc):
    def __init__(self,**std_atmosphere_args):
        self._atm = std_atmosphere(**std_atmosphere_args)
        Calc.__init__(self,UniformFermatEquationsEuclid(self._atm._n,self._atm._dndy))

    @property
    def atm_model(self):
        return self._atm


class CurveNoRefraction(Calc):
    def __init__(self,R0=6370997.0):
        self._R0 = R0
        Calc.__init__(self,UniformFermatEquationsCurve(R0,lambda s,y:1.0,lambda s,y:0.0))

    @property
    def R0(self):
        return self._R0


class FlatNoRefraction(Calc):
    def __init__(self,**std_atmosphere_args):
        Calc.__init__(self,UniformFermatEquationsEuclid(lambda s,y:1.0,lambda s,y:0.0))
























'''


class FermatEquationsPolar(_EulerEquations):
    """Solver for light ray in a 2D Polar geometry.

    This object takes in three user defined functions:  :math:`n(\\theta,r), \\frac{\partial n(\\theta,r)}{\partial\\theta}, \\frac{\partial n(\\theta,r)}{\partial r}`
    and uses these functions to solve Fermat's equations for the path of a light ray.


    """
    def __init__(self,n,dndtheta,dndr,args=()):
        """Intializes the `FermatEquationsPolar` object.

        Parameters
        ----------
        n : callable
            function which returns the index of refraction: :math:`n(\theta,r)`.
        dndx : callable
            function which returns :math:`\\frac{\partial n(\\theta,r)}{\\partial\\theta}`.
        dndy : callable
            function which returns :math:`\\frac{\partial n(\\theta,r)}{\\partial r}`.
        args : array_like, optional
            optional arguments which go into the functions. 

        """
        self._n = n
        self._dndtheta = dndtheta
        self._dndr = dndr
        self._args = args    


    def __call__(self,theta,yin):
        if np.any(np.isnan(yin)):
            raise ValueError
        shape0 = yin.shape
        yin = yin.reshape((2,-1))
        try:
            self._yout[...] = yin[...]
        except ValueError:
            self._yout = yin.copy()
            
        r,dr = yin[0],yin[1]
        n_val = self._n(theta,r,*self._args)
        dndtheta_val = self._dndtheta(theta,r,*self._args)
        dndr_val = self._dndr(theta,r,*self._args)

        r2 = r**2
        dr2 = dr**2

        self._yout[0] = dr
        self._yout[1] = (n_val*r*(r2+2*dr)+(r2+dr2)*(r2*dndr_val-dr*dndtheta_val))/(n_val * r2)
        # yout[1] = ((-dndtheta_val)*dr*(dr2 + r2) + r*(r2*(n_val + dndr_val*r) + dr2*(2*n_val + dndr_val*r)))/(n_val*r2)

        return self._yout.reshape(shape0)



class CurveCalc(object):
    """
    Ray trace calculator for a curved earth.
    """
    def __init__(self,h0=0.0,T0=15.0,P0=101325.0,g=9.8076,dT=None,moist_lapse_rate=False,
                 R0=6370997.0,T_prof=None,dT_prof=None,T_prof_args=(),n_funcs=None):
        """Intializes the `CurveCalc` object.
        
        Parameters
        ----------
        h0: float, optional
            height (in :math:`m`) to determine initial conditions on atmospheric model

        T0: float, optional
            Temperature (in degrees :math:`C`) at `h0`

        P0: float, optional
            Pressure (in :math:`Pa`) at `h0`

        g: float, optional
            gravitational acceleration (in :math:`m/s^2`)

        dT: float, optional
            Temperature lapse rate (in :math:`C/m`) defined as minus the slope of the temperature

        moist_lapse_rate: bool, optional
            Uses the temperature and pressure to calculate the moist lapse rate, use when the humidity is at 100%

        R0: float, optional
            radius of the earth for calculation.

        T_prof: callable, optional
            user defined function, see description. 

        dT_prof: callable, optional
            derivative of `T_prof`, see description.

        T_prof_args: array_like, optional
            optional arguments to pass into `T_prof` and `dT_prof`.

        n_funcs: array_like, optional
            object which contains three callable functions: :math:`n(\theta,r),\frac{\partial n(\theta,r)}{\partial\theta},\frac{\partial n(\theta,r)}{\partial r}` in this order.

        """

        wavelength=545.0
        self._R0 = R0

        if n_funcs is None:
                    
            T0 = max(T0,0)
            e = 611.21*np.exp((18.678-T0/234.5)*(T0/(557.14+T0)))
            T0 += 273
            eps = 0.622
            cpd = 1003.5
            Hv = 2501000
            R = 287.058
            r = eps*e/(P0-e)

            if dT is None:
                if moist_lapse_rate:
                    dT = g*(R*T0**2+Hv*r*T0)/(cpd*R*T0**2+r*eps*Hv**2)
                else:
                    dT = 0.0098
                

            if T_prof is not None and dT_prof is not None:
                T_prof0 = T_prof(h0,*T_prof_args)
                T = lambda r:T0-dT*(r-h0-R0)+(T_prof(r-R0,*T_prof_args)-T_prof0)
                dTdr = lambda r:-dT+dT_prof(r-R0)
            if T_prof is not None:
                T_prof0 = T_prof(h0,*T_prof_args)
                T = lambda r:T0-dT*(r-h0-R0)+(T_prof(r-R0,*T_prof_args)-T_prof0)
                dTdr = lambda r:-dT+(T_prof(r-R0+1.1e-7,*T_prof_args)-T_prof(r-R0-1.1e-7,*T_prof_args))/(2.2e-7)
            else:
                T = lambda r:T0-dT*(r-h0-R0)
                dTdr = lambda r:-dT


            dPdh = lambda r,P:-g*P/(R*T(r))

            sol = solve_ivp(dPdh,(R0+h0,R0-10000),np.array([P0]),atol=1e-7,rtol=1e-13)

            P = sol.y[0,-1]
            sol = solve_ivp(dPdh,(R0-10000,R0+10000),np.array([P]),dense_output=True,atol=1e-7,rtol=1e-13)


            def drhodr(r):
                t = T(r)
                p = sol.sol(r)[0]
                dpdr = -g*p/(R*t)
                return (dpdr*t-dTdr(r)*p)/(R*t**2)

            rho = lambda r:sol.sol(r)[0]/(R*T(r))


            deltan = 0.0002879*(1+0.0000567/wavelength**2)
            n = lambda theta,r:(1+rho(r)*deltan)
            dndr = lambda theta,r:drhodr(r)*deltan
            dndtheta = lambda theta,r:0.0


            self._rho = rho
            self._dT = dT
            self._n = n
            self._P = lambda r:sol.sol(r)[0]
            self._T = T
        else:
            n,dndtheta,dndr = n_funcs

        self._ee = FermatEquationsPolar(n,dndtheta,dndr)

    @property
    def dT(self):
        """ Temperature Lapse rate for this atmospheric model. """
        return self._dT

    def T(self,h):
        """ Temperature function for this atmospheric model. """
        return self._T(self._R0+h)

    def P(self,h):
        """ Pressure function for this atmospheric model. """
        return self._P(self._R0+h)

    def n(self,h):
        """ Index of refraction function for this atmospheric model. """
        return self._n(0.0,self._R0+h)

    def rho(self,h):
        """ density function for this atmospheric model. """
        return self._rho(self._R0+h)

    @property
    def R0(self):
        """ Radius of the earth for this calcuulator. """
        return self._R0

    def solve_ivp(self,d,h,dh=None,alpha=None,**kwargs):
        """

        """
        theta_a = 0
        theta_b = float(d)/self._R0
        
        h = np.asarray(h)

        R_a = self._R0+h

        if alpha is not None:
            dR_a = np.asarray(R_a*np.tan(np.deg2rad(alpha)))
        else:
            dR_a = np.asarray(dh)

        R_a,dR_a = np.broadcast_arrays(R_a,dR_a)


        return self._ee.solve_ivp(theta_a,theta_b,R_a,dR_a,**kwargs)


'''