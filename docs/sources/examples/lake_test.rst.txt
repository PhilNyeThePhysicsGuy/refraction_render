.. _lake_test-label:

Laser Tests Over Cool Lake
--------------------------

In this example we use the refraction calculators to calculate the trajectory of a laser beam over the surface of the earth. In this case we simulate an atmosphere which has cool air near the surface of the earth which creates the effect of bending the laser light down towards the surface of the earth.

:download:`download files <../../examples/zips/lake_test.zip>`

.. literalinclude:: ../../examples/zips/lake_test/lake_test.py
	:linenos:
	:language: python
	:lines: 1-


Instead of using the renderer built into the package we use Matplotlib to plot the actual trajectories of the rays. The model for the atmosphere uses this Temperature profile:

.. image:: ../../examples/example_renders/T_prof.png
   :align: center
   :scale: 20 %

We show the results of a laser at 7 ft above the water level shining light over 10.3 miles to the observers on the other side.

.. image:: ../../examples/example_renders/lake_test.png
   :align: center

The conclusion is that the Laser can be seen at all heights on the opposite side of the lake giving some people the impression that the earth is flat however this conclusion is lacking the sophisticated analysis presented here. 
