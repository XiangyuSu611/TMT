#version 450 core
#include "envmap/spheremap.glsl"

uniform samplerCube u_cubemap;

in vec2 v_uv;
out vec4 out_color;

void main() {
  vec3 samp_vec = sphere_tex_to_world(v_uv);
  out_color = vec4(texture(u_cubemap, samp_vec).xyz, 1.0);
}
