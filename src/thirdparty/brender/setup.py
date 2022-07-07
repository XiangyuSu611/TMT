from distutils.core import setup
setup(
  name = 'brender',
  packages = ['brender'],
  version = '0.1',
  description = 'A simple wrapper for the Blender API (bpy)',
  author = 'Keunhong Park',
  author_email = 'kpar@cs.washington.edu',
  url = 'https://github.com/keunhong/brender',
  download_url = 'https://github.com/keunhong/brender/archive/0.1.tar.gz',
  keywords = ['python', 'brender', 'blender'],
  classifiers = [],
  install_requires=[
    'numpy',
    'scipy',
    'toolbox~=0.2.0',
  ],
  dependency_links=['http://github.com/keunhong/toolbox/tarball/master#egg=package-0.2']
)

