#version 450 core
#include "envmap/spheremap.glsl"
#include "envmap/cubemap.glsl"

uniform sampler2D u_panorama;
uniform int u_cube_face;

in vec2 v_uv;
out vec4 out_color;

void main() {
  vec2 pos = v_uv * 2.0 - 1.0;
  vec3 normal = cubemap_face_to_world(pos, u_cube_face);
  vec2 pano_uv = sphere_world_to_tex(normal);
  vec3 color = texture(u_panorama, pano_uv).xyz;
  out_color = vec4(color, 1.0);
}
