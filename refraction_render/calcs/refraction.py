from scipy.integrate import solve_bvp,solve_ivp
import numpy as np

__all__=["CurveCalc","FlatCalc","FermatEquationsEuclid","FermatEquationsPolar"]

class _EulerEquations(object):
    def __init__(self):
        pass

    def solve_ivp(self,a,b,y0,dy0,**kwargs):
        """Solve initial value problem for light rays.

        Notes
        -----
        The solver can solve the path of an arbitrary number of light rays in one function call
        however the format of the solutions has the y(x) positions stacked on top of the derivatives.

        Parameters
        ----------
        a : scalar
            initial position for the solution of the ray's path.
        b : scalar
            final position for the solution of the ray's path.
        y0 : array_like of shape (n,)
            initial position of the ray's.
        dy0 : array_like of shape (n,)
            initial derivative (with respect to the independent variable) of the ray's trajectory.
        kwargs : optional
            additional arguments to pass into solver, 
            see scipy.integrate.solve_ivp for more details.


        Returns
        -------
        Bunch object with the following fields defined:
        t : ndarray, shape (n_points,)
            Time points.
        y : ndarray, shape (n, n_points)
            Values of the solution at `t`.
        sol : `OdeSolution` or None
            Found solution as `OdeSolution` instance; None if `dense_output` was
            set to False.
        t_events : list of ndarray or None
            Contains for each event type a list of arrays at which an event of
            that type event was detected. None if `events` was None.
        nfev : int
            Number of evaluations of the right-hand side.
        njev : int
            Number of evaluations of the Jacobian.
        nlu : int
            Number of LU decompositions.
        status : int
            Reason for algorithm termination:
                * -1: Integration step failed.
                *  0: The solver successfully reached the end of `tspan`.
                *  1: A termination event occurred.
        message : string
            Human-readable description of the termination reason.
        success : bool
            True if the solver reached the interval end or a termination event
            occurred (``status >= 0``).


        """
        y0 = np.array([y0,dy0])
        self._yout = np.zeros_like(y0.reshape((2,-1)))
        return solve_ivp(self,(a,b),y0.ravel(),**kwargs)

class FermatEquationsEuclid(_EulerEquations):
    """Solver for light ray in a 2D Euclidian geometry.

    This object takes in three user defined functions:  :math:`n(x,y), dn/dx, dn/dy`
    and uses these functions to solve Fermat's equations for the path of a light ray.


    """
    def __init__(self,n,dndx,dndy,args=()):
        """Intializes the `FermatEquationsEuclid` object.

        Parameters
        ----------
        n : callable
            function which returns the index of refraction :math:`n(x,y)`.
        dndx : callable
            function which returns :math:`\\frac{\partial n(x,y)}{\partial x}`.
        dndy : callable
            function which returns :math:`\\frac{\partial n(x,y)}{\partial y}`.
        args : array_like, optional
            optional arguments which go into the functions. 

        """
        self._n = n
        self._dndx = dndx
        self._dndy = dndy
        self._args = args

    def __call__(self,x,yin): # object(args,...)
        shape0 = yin.shape
        yin = yin.reshape((2,-1))
        try:
            self._yout[...] = yin[...]
        except ValueError:
            self._yout = yin.copy()

        y,dydx = yin[0],yin[1]

        n_val = self._n(x,y,*self._args)
        dndx_val = self._dndx(x,y,*self._args)
        dndy_val = self._dndy(x,y,*self._args)

        self._yout[0] = dydx
        self._yout[1] = (1+dydx**2)*(dndy_val-dydx*dndx_val)/n_val

        return self._yout.reshape(shape0)

class FermatEquationsPolar(_EulerEquations):
    def __init__(self,n,dndtheta,dndr,args=()):
        """Intializes the `FermatEquationsPolar` object.

        Parameters
        ----------
        n : callable
            function which returns the index of refraction: :math:`n(\theta,r)`.
        dndx : callable
            function which returns :math:`\\frac{\partial n(\theta,r)}{\partial\theta}`.
        dndy : callable
            function which returns :math:`\\frac{\partial n(\theta,r)}{\partial r}`.
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
    def __init__(self,T0=15.0,P0=101325.0,h0=0.0,g=9.8076,dT=None,moist_lapse_rate=False,
                wavelength=545,dT_prof=None,T_prof=None,phi=None,T_prof_args=(),n_funcs=None):


        if phi is None:
            R0 = 6370997.0
            self._R0 = 6370997.0
        else:
            R0 = effective_curvature(phi)
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
                T_prof0 = T_prof(h0+R0,*T_prof_args)
                T = lambda r:T0-dT*(r-(h0+R0))+(T_prof(r-R0,*T_prof_args)-T_prof0)
                dTdr = lambda r:-dT+dT_prof(r-h0)
            if T_prof is not None:
                T_prof0 = T_prof(h0+R0,*T_prof_args)
                T = lambda r:T0-dT*(r-(h0+R0))+(T_prof(r-R0,*T_prof_args)-T_prof0)
                dTdr = lambda r:-dT+(T_prof((r-R0)+1.1e-7,*T_prof_args)-T_prof((r-R0)-1.1e-7,*T_prof_args))/(2.2e-7)
            else:
                T = lambda r:T0-dT*(r-(h0+R0))
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
            n,dndr,dndtheta = n_funcs

        self._ee = FermatEquationsPolar(n,dndtheta,dndr)

    @property
    def dT(self):
        return self._dT

    def T(self,h):
        return self._T(self._R0+h)

    def P(self,h):
        return self._P(self._R0+h)

    def n(self,h):
        return self._n(0.0,self._R0+h)

    def rho(self,h):
        return self._rho(self._R0+h)

    @property
    def R0(self):
        return self._R0

    def solve_ivp(self,d,h,dh=None,alpha=None,R0=None,**kwargs):
        theta_a = np.pi/2
        theta_b = np.pi/2+d/self._R0
        h = np.array(h)

        if R0 is None:
            R_a = (self.R0+h)
        else:
            R_a = R0+h
            
        if alpha is not None:
            dR_a = np.array(R_a*np.tan(np.deg2rad(alpha)))
        else:
            dR_a = np.array(dh)

        if R_a.ndim == 0 and dR_a.ndim > 0:
            R_a = np.ones_like(dR_a)*R_a
        elif R_a.ndim > 0 and dR_a.ndim == 0:
            dR_a = np.ones_like(R_a)*dR_a
        elif R_a.ndim > 0 and dR_a.ndim > 0:
            if len(R_a) != len(dR_a):
                raise ValueError("number of initial positions and slops must be equal.")

        sol = self._ee.solve_ivp(theta_a,theta_b,R_a,dR_a,**kwargs)

        return sol

class FlatCalc(object):
    def __init__(self,T0=15.0,P0=101325.0,h0=0.0,g=9.81,dT=None,moist_lapse_rate=False,
                wavelength=545,T_prof=None,dT_prof=None,T_prof_args=(),n_funcs=None):

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
                T_prof0 = T_prof(h0+R0,*T_prof_args)
                T = lambda r:T0-dT*(r-(h0+R0))+(T_prof(r-R0,*T_prof_args)-T_prof0)
                dTdr = lambda r:-dT+dT_prof(r-h0)
            if T_prof is not None:
                T = lambda r:T0-dT*(r-h0)+T_prof(r,*T_prof_args)
                dTdr = lambda r:-dT+(T_prof(r+1.1e-7,*T_prof_args)-T_prof(r-1.1e-7,*T_prof_args))/(2.2e-7)
            else:
                T = lambda r:T0-dT*(r-h0)
                dTdr = lambda r:-dT

            dPdh = lambda r,P:-g*P/(R*T(r))

            sol = solve_ivp(dPdh,(h0,-10000),np.array([P0]))

            P = sol.y[0,-1]
            sol = solve_ivp(dPdh,(-10000,10000),np.array([P]),dense_output=True)

            def drhody(r):
                t = T(r)
                p = sol.sol(r)[0]
                dpdr = -g*p/(R*t)
                return (dpdr*t-dTdr(r)*p)/(R*t**2)

            rho = lambda r:sol.sol(r)[0]/(R*T(r))


            deltan = 0.0002879*(1+0.0000567/wavelength**2)
            n = lambda theta,r:(1+rho(r)*deltan)
            dndr = lambda theta,r:drhodr(r)*deltan
            dndtheta = lambda theta,r:0.0


            deltan = 0.0002879*(1+0.0000567/wavelength**2)
            n = lambda theta,r:(1+rho(r)*deltan)
            dndy = lambda theta,r:drhody(r)*deltan
            dndx = lambda theta,r:0.0
            
            self._rho = rho
            self._dT = dT
            self._n = n
            self._P = lambda h:P(h)[0]
            self._T = T
        else:
            n,dndy,dndx = n_funcs




        self._ee = FermatEquationsEuclid(n,dndx,dndy)


    @property
    def R0(self):
        return 0

    @property
    def dT(self):
        return self._dT

    def T(self,h):
        return self._T(h)

    def P(self,h):
        return self._P(h)

    def n(self,h):
        return self._n(0,h)

    def solve_bvp(self,d,ha,hb):
        return self._ee.solve_bvp(0,d,ha,hb)

    def solve_ivp(self,d,h,dh=None,alpha=None,**kwargs):
        if alpha is not None:
            dh = np.tan(np.deg2rad(alpha))
        
        h = np.array(h)
        if h.ndim == 0 and dh.ndim > 0:
            h = np.ones_like(dh)*h
        elif h.ndim > 0 and dh.ndim == 0:
            dh = np.ones_like(h)*dh
        elif h.ndim > 0 and dh.ndim > 0:
            if len(h) != len(dh):
                raise ValueError("number of initial positions and slops must be equal.")


        return self._ee.solve_ivp(0,d,h,dh,**kwargs)


