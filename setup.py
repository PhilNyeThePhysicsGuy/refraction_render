def configuration(parent_package='', top_path=None):
	from numpy.distutils.misc_util import Configuration
	config = Configuration(None, parent_package, top_path)
	config.set_options(
	assume_default_configuration=True,
	delegate_options_to_subpackages=True,
	quiet=True)

	config.add_subpackage('refraction_render')

	return config


def setup_package():
	try:
		import numpy
	except:
		raise ImportError("build requires numpy for fortran extensions")

	metadata = dict(
		name='refraction_render',
		version="1.0.0",
		maintainer="Phillip Weinberg",
		download_url="https://github.com/PhilNyeThePhysicsGuy/refraction_render.git",
		license='BSD',
		platforms=["Unix","Windows"]
	)

	from numpy.distutils.core import setup
	metadata['configuration'] = configuration

	setup(**metadata)


if __name__ == '__main__':
	setup_package()




