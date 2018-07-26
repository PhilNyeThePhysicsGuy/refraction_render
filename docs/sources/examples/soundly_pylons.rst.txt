.. _soundly_pylons-label:

Lake Pontchartrain Pylons
-------------------------

:download:`download files <../../examples/zips/soundly_pylons.zip>`

So there is a YouTuber named Soundly who makes observations of various structures over Lake Pontchartrain. This lake has a bunch of structures which line along a perfectly straight line path across the water which makes them perfect for showing the curvature of the earth!

In this example shows how we can use the render to model the observation of some power lines which go across a significant portion of the lake:

.. literalinclude:: ../../examples/zips/soundly_pylons/soundly_pylons.py
	:linenos:
	:language: python
	:lines: 1-


Now to compare reality to our render! Here is one of Soundly's original images of the power lines:

.. image:: ../../examples/images/pylons.jpg
   :align: center
   :scale: 50 %

Here is a render trying to reproduce that result.

.. image:: ../../examples/example_renders/soundly_pylons.png
   :align: center
   :scale: 10 %

One interesting effect which can't really be seen is the inferior mirage which is creating a distortion of the bottom parts of the pylons. Here is a zoomed in view of the actual pylons:

.. image:: ../../examples/images/pylons_zoom.jpg
   :align: center
   :scale: 50 %

Compared to the rendered model zoomed in:

.. image:: ../../examples/example_renders/soundly_pylons_zoom.png
   :align: center
   :scale: 10 %



