# refraction rendering

This is a python package which is a very simple ray tracer which allows the user to ray trace simple topographical data as well as images in the presence of atmospheric refraction. The package includes tools to calculate the [standard atmospheric model](https://en.wikipedia.org/wiki/International_Standard_Atmosphere) for an arbitrary temperature profile. From this profile it can generate an index of refraction profile. The Rays trajectories are solved for using [Fermat's principle](https://en.wikipedia.org/wiki/Fermat%27s_principle).

This is a python package which uses SciPy, NumPy, Numba, pyproj and six so make sure these packages are installed before installing this code.

To install simply download the code and from the top directory run `python setup.py install`. 

Check out the documentation at https://philnyethephysicsguy.github.io/refraction_render/


