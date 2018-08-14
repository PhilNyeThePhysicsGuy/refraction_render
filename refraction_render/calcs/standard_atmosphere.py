from scipy.integrate import solve_bvp,solve_ivp
from scipy.misc import derivative
import numpy as np


__all__=["std_atmosphere"]

class std_atmosphere(object):
    """Object which calculates the standard atmospheric model. 

    """
    def __init__(self,h0=0.0,T0=15.0,P0=101325.0,g=9.81,dT=None,moist_lapse_rate=False,
                 T_prof=None,dT_prof=None,T_prof_args=()):
        """Intializes the `std_atmosphere` object.

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

        T_prof: callable, optional
            user defined function, see description. 

        dT_prof: callable, optional
            derivative of `T_prof`, see description.

        T_prof_args: array_like, optional
            optional arguments to pass into `T_prof` and `dT_prof`.

        """

        wavelength=0.545
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
            T = lambda h:T0-dT*(h-h0)+(T_prof(h,*T_prof_args)-T_prof0)
            dTdr = lambda h:-dT+dT_prof(h)
        if T_prof is not None:
            T_prof0 = T_prof(h0,*T_prof_args)
            T = lambda h:T0-dT*(h-h0)+(T_prof(h,*T_prof_args)-T_prof0)
            dTdr = lambda h:-dT+derivative(T_prof,h,args=T_prof_args,dx=1.1e-7)
        else:
            T = lambda h:T0-dT*(h-h0)
            dTdr = lambda h:-dT

        dPdh = lambda h,P:-g*P/(R*T(h))

        sol = solve_ivp(dPdh,(h0,-10000),np.array([P0]))

        P = sol.y[0,-1]
        sol = solve_ivp(dPdh,(-10000,10000),np.array([P]),dense_output=True)

        def drhody(h):
            t = T(h)
            p = sol.sol(h)[0]
            dpdr = -g*p/(R*t)
            return (dpdr*t-dTdr(h)*p)/(R*t**2)

        rho = lambda h:sol.sol(h)[0]/(R*T(h))


        deltan = 0.0002879*(1+0.0000567/wavelength**2)
        n = lambda s,h:(1+rho(h)*deltan)
        dndr = lambda s,h:drhodr(h)*deltan
        dnds = lambda s,h:0.0


        deltan = 0.0002879*(1+0.0000567/wavelength**2)
        self._n = lambda s,h:(1+rho(h)*deltan)
        self._dndy = lambda s,h:drhody(h)*deltan
        self._dndx = lambda s,h:0.0
        
        self._rho = rho
        self._dT = dT
        self._n = n
        self._P = lambda h:P(h)[0]
        self._T = T
        self._dTdh = dTdr
            

    @property
    def dT(self):
        """ Temperature Lapse rate for this atmospheric model. """
        return self._dT

    def T(self,h):
        """ Temperature function for this atmospheric model. """
        return self._T(h)-273

    def P(self,h):
        """ Pressure function for this atmospheric model. """
        return self._P(h)

    def n(self,h):
        """ Index of refraction function for this atmospheric model. """
        return self._n(0,h)

    def rho(self,h):
        """ density function for this atmospheric model. """
        return self._rho(h)

    def dTdh(self,h):
        """ density function for this atmospheric model. """
        return self._dTdh(h)

    def k(self,h):
        """ Curvature of light rays at given height."""
        return 5.03*self._P(h)*(0.0343+self._dTdh(h))/self._T(h)**2