from scipy.integrate import solve_bvp,solve_ivp
from scipy.misc import derivative
from scipy.interpolate import interp1d,RectBivariateSpline
import numpy as np


__all__=["std_atmosphere","atmospheric_corridor"]

class std_atmosphere(object):
    """Object which calculates the standard atmospheric model. 

    """
    def __init__(self,h0=0.0,T0=15.0,P0=101325.0,g=9.81,dT=None,wavelength=0.545,moist_lapse_rate=False,
                 T_prof=None,dT_prof=None,T_prof_args=(),h_min=-10000,h_max=10000):
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

        wavelength: float, optional
            wavelength of light (in :math:`\\mu m`) used calculate the index of refraction 

        moist_lapse_rate: bool, optional
            Uses the temperature and pressure to calculate the moist lapse rate, use when the humidity is at 100%

        T_prof: callable, optional
            user defined function, see description. 

        dT_prof: callable, optional
            derivative of `T_prof`, see description.

        T_prof_args: array_like, optional
            optional arguments to pass into `T_prof` and `dT_prof`.

        h_min: float, optional
            minimal elevation in meters to calculate atmospheric profile, default values is -10000

        h_max: float, optional
            maximal elevation in meters to calculate atmospheric profile, default values is 10000

        """

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

        sol = solve_ivp(dPdh,(h0,h_min),np.array([P0]))

        P = sol.y[0,-1]
        sol = solve_ivp(dPdh,(h_min,h_max),np.array([P]),dense_output=True)

        if wavelength < 0.23  or wavelength > 1.69:
              warnings.warm("Cauchy Equation used to calculate despersion does not work well beyond the visible spetrum. ")
          
        deltan = (0.05792105/(238.0185-wavelength**(-2)) + 0.00167917/(57.362-wavelength**(-2)))
        self._n = lambda s,h:1+deltan*np.squeeze(sol.sol(h))/(R*T(h))
        
        self._rho = lambda h:np.squeeze(sol.sol(h))/(R*T(h))
        self._dT = dT
        self._P = lambda h:np.squeeze(sol.sol(h))
        self._T = T
        self._dTdh = dTdr

        self._g = 9.81
        self._deltan = deltan

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

    @property
    def R(self):
        """Ideal gas constant for the atmosphere"""
        return 287.058
    
    @property
    def g(self):
        """gravitational acceleration."""
        return self._g
    
    @property
    def deltan(self):
        """scale factor used to calculate the index of refraction of air. """
        return self._deltan

    def _f(self,s,h):

        t = self._T(h)
        p = np.squeeze(self._P(h))
        dpdr = -self.g*p/(self.R*t)

        rho = p/(self.R*t)
        drhody = (dpdr*t-self._dTdh(h)*p)/(self.R*t**2)

        return (1+self.deltan*rho),self.deltan*drhody


class atmospheric_corridor(object):
    """Object which calculates the standard atmospheric model. 

    """
    def __init__(self,T_prof,dist_vals,h0=0.0,P0=101325.0,g=9.81,
        wavelength=0.545,h_vals=None,T_prof_args=()):
        """Intializes the `std_atmosphere` object.

        Parameters
        ----------

        T_prof: callable
            user defined function for the temperature as a function of height. in Deg. C

        h0: float, optional
            height (in :math:`m`) to determine initial conditions on atmospheric model

        P0: float, optional
            Pressure (in :math:`Pa`) at `h0`

        g: float, optional
            gravitational acceleration (in :math:`m/s^2`)

        wavelength: float, optional
            wavelength of light (in :math:`\\mu m`) used calculate the index of refraction 

        T_prof_args: array_like, optional
            optional arguments to pass into `T_prof` and `dT_prof`.

        """

        R =  287.058
        T = lambda h:T_prof(h,*T_prof_args)+273

        dPdh = lambda h,P:-g*P/(R*T(h))
        # def dPdh(h,P):
        #     print(g,P.shape,R,h.shape)
        #     return -g*P/(R*T(h))

        dist_vals = np.asarray(dist_vals)
        shape = dist_vals.shape
        P0 = np.broadcast_to(P0,shape)
        sol = solve_ivp(dPdh,(h0,-10001),P0)
        
        if h_vals is None:
            h_vals = np.hstack([np.linspace(-10000,0,1000),
                                np.linspace(0,300,3001),
                                np.linspace(400,900,5),
                                np.linspace(1000,10000,1000)
                                ])

        h_vals = np.unique(h_vals)
        sol = solve_ivp(dPdh,(-10001,10001),sol.y[:,-1],t_eval = h_vals)

        Ps = sol.y


        if wavelength < 0.23  or wavelength > 1.69:
              warnings.warm("Cauchy Equation used to calculate despersion does not work well beyond the visible spetrum. ")
          
        self._deltan = (0.05792105/(238.0185-wavelength**(-2)) + 0.00167917/(57.362-wavelength**(-2)))


        Ts = np.hstack([T(h).reshape((-1,1)) for h in h_vals])
        rhos = Ps/(R*Ts)

        self._dist_vals = dist_vals
        self._h_vals = h_vals
        self._rho_interp = RectBivariateSpline(dist_vals,h_vals,rhos)
        self._T_interp = RectBivariateSpline(dist_vals,h_vals,Ts)
        self._P_interp = RectBivariateSpline(dist_vals,h_vals,Ps)
        self._n_interp = RectBivariateSpline(dist_vals,h_vals,1+self._deltan*rhos)


    def T(self,s,h):
        return self._T_interp(s,h,grid=False) - 273

    def P(self,s,h):
        return self._P_interp(s,h,grid=False)

    def rho(self,s,h):
        return self._rho_interp(s,h,grid=False)

    def _f(self,s,h):
        s,h = np.broadcast_arrays(s,h)

        s = np.atleast_1d(s)
        h = np.atleast_1d(h)

        s_m = np.logical_or(s < self._dist_vals[0], s > self._dist_vals[-1])
        h_m = np.logical_or(h < self._h_vals[0], h > self._h_vals[-1])

        n = self._n_interp(s,h,grid=False)
        dndy = self._n_interp(s,h,dy=1,grid=False)
        dnds = self._n_interp(s,h,dx=1,grid=False)

        dnds[s_m] = 0
        dndy[h_m] = 0

        return n,dndy,dnds


