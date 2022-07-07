# rendkit
Rendkit is a simple and extensible rendering library. It currently uses [vispy](https://github.com/vispy) for OpenGL bindings.

Rendkit provides several reflectance models in the form of shaders. These include

 * The Phong model
 * The spatially-varying microfacet model from Aitalla et al.
 * A basic single-colored material.

The following post-processing options are also available

 * Naive SSAA antialiasing.
 * Gamma correction.

Rendkit also provides a flexible scene descriptor language called JSON Scene Descriptor (JSD). It's just a JSON file with a specific convention that rendkit understands.

## Dependencies
 * Vispy
 * Numpy
