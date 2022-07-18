
vec3 dualp_tex_to_world(vec2 uv, int hemisphere, float b) {
  float b2 = b * b;
  float s = uv.s - 0.5;
  float t = uv.t - 0.5;
  float s2 = s * s;
  float t2 = t * t;
  float denom = (4.0*b2*s2 + 4.0*b2*t2 + 1.0);
  vec3 v;
  if (hemisphere == 0) {
    v.x = -(4.0 * b * s) / denom;
    v.z = -(4.0 * b * t) / denom;
    v.y = sqrt(1.0 - v.x*v.x - v.z*v.z);
  } else if (hemisphere == 1) {
    v.x = (4.0 * b * s) / denom;
    v.z = (4.0 * b * t) / denom;
    v.y = -sqrt(1.0 - v.x*v.x - v.z*v.z);
  }
  return normalize(v);
}


vec2 dualp_world_to_tex(vec3 world, float b) {
  vec2 uv;
  if (world.y > 0) {
    uv.s = -world.x / (2.0 * b * (1.0 + world.y)) + 0.5;
    uv.t = -world.z / (2.0 * b * (1.0 + world.y)) + 0.5;
  } else {
    uv.s = world.x / (2.0 * b * (1.0 - world.y)) + 0.5;
    uv.t = world.z / (2.0 * b * (1.0 - world.y)) + 0.5;
  }
  return uv;
}
