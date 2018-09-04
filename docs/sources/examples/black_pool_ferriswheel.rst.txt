.. _blackpool_ferriswheel-label:

Black Pool Ferris wheel from Barrow in Furness
----------------------------------------------

back to :ref:`examples`

In this example we try to model an observation done by Ranty Flat Earth 

:download:`download files <../../examples/zips/blackpool_ferriswheel.zip>`

.. literalinclude:: ../../examples/zips/blackpool_ferriswheel/blackpool_ferriswheel.py
	:linenos:
	:language: python
	:lines: 1-


Just like in the example: :ref:`lake_test-label`, The model for the atmosphere uses this temperature profile which has cool air near the water's surface. This is because when there is no direct sun on the surface of the water, the air temperature near the surface is primarily driven by the temperature of the water. If the water temperature is lower than the air temperature the air near the surface will be cooler than the air above. For this model we use the following temperature profile:

.. image:: ../../examples/example_renders/T_prof_blackpool.png
   :align: center
   :scale: 20 %

That can be used to calculate the air density as a function of height

.. image:: ../../examples/example_renders/rho_prof_blackpool.png
   :align: center
   :scale: 20 %

This model leads to the following result for a render of the Ferris wheel:

.. image:: ../../examples/example_renders/ferris_wheel.png
   :align: center
   :scale: 50 %

Compare this to a still image grabbed from the video of the observation. 

.. image:: ../../examples/images/blackpool_ferriswheel.png
   :align: center
   :scale: 50 %

Notice there that there is a bit of compression in the image. This is because the temperature gradient at the low level for the atmosphere. The conclusion from this simulation is that the conditions for Ranty's observations are not impossible on a globe and therefore the observation is not impossible on the globe. 

back to :ref:`examples`
