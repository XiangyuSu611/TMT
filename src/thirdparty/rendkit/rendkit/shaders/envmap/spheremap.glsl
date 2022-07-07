#include "utils/math.glsl"


vec3 sphere_tex_to_world(vec2 uv) {
  float theta = (uv.s * 2.0 - 1.0) * M_PI - M_PI / 2;
  float phi = (uv.t * 2.0 - 1.0) * M_PI / 2.0;
  vec3 world;
  world.x = cos(phi) * cos(theta);
  world.y = sin(phi);
  world.z = cos(phi) * sin(theta);
  return world;
}


vec2 sphere_world_to_spherical(vec3 world) {
  float theta = atan(world.x, world.z);
  float phi = acos(world.y);
  return vec2(theta, phi);
}


vec2 sphere_world_to_tex(vec3 world) {
  vec2 theta_phi = sphere_world_to_spherical(world);
  theta_phi.s = ((theta_phi.s / M_PI) + 1.0)/2.0;
  theta_phi.t = (theta_phi.t / M_PI);
  return theta_phi;
}
