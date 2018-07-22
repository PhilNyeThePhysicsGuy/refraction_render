"""
=============================================
Calcs module (:mod:`refraction_render.calcs`)
=============================================

classes for solving Fermat's principle:
---------------------------------------

.. currentmodule:: refraction_render.calcs

.. autosummary::
   :toctree: generated/

   FermatEquationsEuclid
   FermatEquationsCurve
   UniformFermatEquationsEuclid
   UniformFermatEquationsCurve

classes for solving rays in atmospheric model:
----------------------------------------------

.. autosummary::
   :toctree: generated/

   Calc
   CurveCalc
   CurveNoRefraction
   FlatCalc
   FlatNoRefraction



"""

from .calculators import *
from .euler_equations import *
from .standard_atmosphere import *