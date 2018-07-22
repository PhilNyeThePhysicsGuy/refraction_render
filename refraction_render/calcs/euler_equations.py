from scipy.integrate import solve_bvp,solve_ivp
import numpy as np

__all__=["FermatEquationsEuclid","UniformFermatEquationsEuclid",
		 "FermatEquationsCurve","UniformFermatEquationsCurve","FermatEquations"]


class FermatEquations(object):
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
        y0 = np.vstack((y0,dy0))
        self._yout = np.zeros_like(y0)
        return solve_ivp(self,(a,b),y0.ravel(),**kwargs)


class FermatEquationsEuclid(FermatEquations):
    """Solver for light ray in a 2D Euclidian geometry for :math:`n=n(x,y)`.

    This object takes in three user defined functions:  :math:`n(x,y), \\frac{\\partial n(x,y)}{\\partial x}, \\frac{\\partial n(x,y)}{\\partial y}`
    and uses these functions to solve Fermat's equations for the path of a light ray. This is only really useful


    """
    def __init__(self,n,dndx,dndy,args=()):
        """Intializes the `FermatEquationsEuclid` object.

        Parameters
        ----------
        n : callable
            function which returns the index of refraction :math:`n(x,y)`.
        dndx : callable
            function which returns :math:`\\frac{\\partial n(x,y)}{\\partial x}`.
        dndy : callable
            function which returns :math:`\\frac{\\partial n(x,y)}{\\partial y}`.
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


class FermatEquationsCurve(FermatEquations):
    """Solver for light ray in a 2D Polar geometry for :math:`n=n(s,y)`.

    This object takes in three user defined functions:  :math:`n(s,y), \\frac{\\partial n(s,y)}{\\partial s}, \\frac{\\partial n(s,y)}{\\partial y}`
    and uses these functions to solve Fermat's equations for the path of a light ray. Here :math:`s=R0\theta` and :math:`y=r-R0` in order to make 
    this equation comparable to the Euclidian geometric. 


    """
    def __init__(self,R0,n,dndx,dndy,args=()):
        """Intializes the `FermatEquationsCurve` object.

        Parameters
        ----------
        n : callable
            function which returns the index of refraction :math:`n(s,y)`.
        dnds : callable
            function which returns :math:`\\frac{\\partial n(s,y)}{\\partial s}`.
        dndy : callable
            function which returns :math:`\\frac{\\partial n(s,y)}{\\partial y}`.
        args : array_like, optional
            optional arguments which go into the functions. 

        """
        self._n = n
        self._dnds = dnds
        self._dndy = dndy
        self._args = args  
        self._R0 = R0 


    def __call__(self,s,yin):
        shape0 = yin.shape
        yin = yin.reshape((2,-1))
        try:
            self._yout[...] = yin[...]
        except ValueError:
            self._yout = yin.copy()

        y,dyds = yin[0],yin[1]

        n_val = self._n(s,y,*self._args)
        dnds_val = self._dndx(s,y,*self._args)
        dndy_val = self._dndy(s,y,*self._args)
        R02 = self._R0**2
        R0 = self._R0

        self._yout[0] = dyds
        self._yout[1] = (-2*dyds*R02*dnds_val - ((-1 + dyds)*R0 - y)*((1 + dyds)*R0 + y)*dndy_val + (R0 + y)*n_val)/(R02*n_val)

        return self._yout.reshape(shape0)


class UniformFermatEquationsEuclid(FermatEquations):
    """Solver for light ray in a 2D Euclidian geometry with :math:`n=n(y)`.

    This object takes in three user defined functions:  :math:`n(y), \\frac{\\partial n(y)}{\\partial y}`
    and uses these functions to solve Fermat's equations for the path of a light ray. 

    """
    def __init__(self,n,dndy,args=()):
        """Intializes the `FermatEquationsEuclid` object.

        Parameters
        ----------
        n : callable
            function which returns the index of refraction :math:`n(y)`.
        dndy : callable
            function which returns :math:`\\frac{\\partial n(y)}{\\partial y}`.
        args : array_like, optional
            optional arguments which go into the functions. 

        """
        self._n = n
        self._dndy = dndy
        self._args = args

    def __call__(self,s,yin): # object(args,...)
        shape0 = yin.shape
        yin = yin.reshape((2,-1))
        try:
            self._yout[...] = yin[...]
        except ValueError:
            self._yout = yin.copy()

        y,dyds = yin[0],yin[1]

        n_val = self._n(s,y,*self._args)
        dndy_val = self._dndy(s,y,*self._args)

        self._yout[0] = dyds
        self._yout[1] = (1+dyds**2)*dndy_val/n_val

        return self._yout.reshape(shape0)


class UniformFermatEquationsCurve(FermatEquations):
    """Solver for light ray in a 2D Polar geometry with :math:`n=n(y)`.

    This object takes in three user defined functions:  :math:`n(y), \\frac{\\partial n(y)}{\\partial y}`
    and uses these functions to solve Fermat's equations for the path of a light ray.


    """
    def __init__(self,R0,n,dndy,args=()):
        """Intializes the `FermatEquationsCurve` object.

        Parameters
        ----------
        n : callable
            function which returns the index of refraction :math:`n(y)`.
        dndy : callable
            function which returns :math:`\\frac{\\partial n(x,y)}{\\partial y}`.
        args : array_like, optional
            optional arguments which go into the functions. 

        """
        self._n = n
        self._dndy = dndy
        self._args = args 
        self._R0 = R0   


    def __call__(self,s,yin):
        shape0 = yin.shape
        yin = yin.reshape((2,-1))
        try:
            self._yout[...] = yin[...]
        except ValueError:
            self._yout = yin.copy()

        y,dyds = yin[0],yin[1]

        n_val = self._n(s,y,*self._args)
        dndy_val = self._dndy(s,y,*self._args)
        R02 = self._R0**2
        R0 = self._R0

        self._yout[0] = dyds
        self._yout[1] = ((R0+y)*n_val-((dyds-1)*R0-y)*(R0*(dyds+1)+y)*dndy_val)/(R02*n_val)

        return self._yout.reshape(shape0)
