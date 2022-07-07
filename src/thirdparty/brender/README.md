# Brender
A small Blender API (bpy) wrapper.


## Introduction

This Python package implements a simple wrapper around the Blender API. It tries to automate the weird stateful API that Blender has but some quirks may leak.

This package implements an incredibly leaky abstraction which means you may still need to understand how the Blender API works.


## Prerequites

You will need to either install this package in the Blender Python interpreter (untested), or install the Blender API package (bpy) as a Python library. Please see https://pypi.org/project/bpy/.




## Example Usage.

```python
  
  import brender
  from imageio import imwrite

  REND_SHAPE = (500, 500)  # HxW
  app = brender.Brender()
  app.init()
  scene = brender.Scene(app, shape=_REND_SHAPE, aa_samples=196)
  scene.set_envmap(_ENVMAP_PATH, scale=5.0)

  # Initialize Camera.
  cam_to_world = np.array([
    [ 0.96592583, -0.12940952,  0.22414387,  0.56035967]
    [ 0.        ,  0.8660254 ,  0.5       ,  1.25      ]
    [-0.25881905, -0.48296291,  0.8365163 ,  2.09129076]
    [ 0.        ,  0.        ,  0.        ,  1.        ]])

  camera = brender.CalibratedCamera(scene, cam_to_world, fov)
  scene.set_active_camera(camera)

  with scene.select():
      mesh = brender.mesh.Monkey(position=(0, 0, 0))
      mesh.enable_smooth_shading()

  rendered_image = scene.render_to_array(format='png')
  imwrite('/tmp/test.png', rendered_image)
```
