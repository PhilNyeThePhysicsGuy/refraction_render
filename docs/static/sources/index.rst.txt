.. refraction_render documentation master file, created by
   sphinx-quickstart on Wed Jun 27 18:42:53 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _refrac: http://www.python.org/

Welcome to refraction_render's documentation!
=============================================

.. |logo1| image:: soundly.gif
   :align: middle
.. |logo2| image:: stenaline.gif
   :align: top
.. |turbine| image:: sphere_composite.png
   :scale: 8%
   :align: middle


+---------+---------+
| |logo1| | |logo2| |
+---------+---------+
|       |turbine|   |
+-------------------+

This is a python package which is a very simple ray tracer which allows the user to ray trace simple topographical data as well as images in the presence of atmospheric refraction. The package includes tools to calculate the `standard atmospheric model <https://en.wikipedia.org/wiki/International_Standard_Atmosphere>`_ for an arbitrary temperature profile. From this profile it can generate an index of refraction profile. The Rays trajectories are solved for using `Fermat's principle <https://en.wikipedia.org/wiki/Fermat%27s_principle>`_.

This is a python package which uses SciPy, NumPy, Numba, pyproj, six, and tqdm so make sure these packages are installed before installing this code. Another useful package is gdal, this is not a requirement to run this code however it is sueful for loading topographical data into python. 

To install simply download the code and from the top directory run ``python setup.py install``. To record the installation directory add the option ``--record <file_name>`` to record the installation path to a file. On unix systems one can easily uninstall the package by running `cat <file_name> | xargs rm -rf`. 


refraction_render API reference
===============================

.. toctree::
    :maxdepth: 1
    
    calcs
    renderers
    misc

Example Scripts
===============

.. toctree::
    :maxdepth: 1
    
    Examples
