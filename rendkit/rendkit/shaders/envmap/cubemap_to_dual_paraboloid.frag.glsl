#version 450 core
#include "envmap/dual_paraboloid.glsl"

uniform samplerCube u_cubemap;
uniform int u_hemisphere;

in vec2 v_uv;
out vec4 out_color;

void main() {
  vec3 samp_vec = dualp_tex_to_world(v_uv, u_hemisphere, 1.2);
  out_color = vec4(texture(u_cubemap, samp_vec).xyz, 1.0);
}
