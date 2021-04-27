:orphan:

.. _lake_test-label:

Laser Tests Over Cool Lake
--------------------------

back to :ref:`examples`

In this example we use the refraction calculators to calculate the trajectory of a laser beam over the surface of the earth. In this case we simulate an atmosphere which has cool air near the surface of the earth which creates the effect of bending the laser light down towards the surface of the earth.

:download:`download files <../../examples/zips/lake_test.zip>`

.. literalinclude:: ../../examples/zips/lake_test/lake_test.py
	:linenos:
	:language: python
	:lines: 1-


Instead of using the renderer built into the package we use Matplotlib to plot the actual trajectories of the rays. The model for the atmosphere uses this Temperature profile:

.. image:: ../../examples/example_renders/T_prof_lake_test.png
   :align: center
   :scale: 20 %

That can be used to calculate the air density as a function of height

.. image:: ../../examples/example_renders/rho_prof_lake_test.png
   :align: center
   :scale: 20 %

which we can then convert into an index of refraction using the refractivity. We show the results of a simulation of various light rays (individual lines in the image below) coming out of a source (the laser) at 7 ft above the water level shining light over 10.3 miles to the observers on the other side. The ray angles are spaced out to represent a guasian distribution of a width of 0.02 degrees which is supposed to simulate the intensity of the beam itself, therefore from the image below the closer together the lines appear the more indense the beam is. The beam is angled down to get the maximum intensity reaching the other side of the lake. 

.. image:: ../../examples/example_renders/lake_test.png
   :align: center
   :scale: 10 %

The conclusion from this simulation is that when there is a layer of cool air above the surface of the earth the Laser light is bent downwards which means it can be seen at all heights on the opposite side of the lake with the intensity of the light depending on the height. These results have giving some people the impression that the earth is flat however this conclusion is lacking the sophisticated analysis presented here. 

back to :ref:`examples`
