#include "utils/sampling.glsl"


float compute_shadow(vec3 world_pos, vec4 shadow_pos, sampler2DShadow depth) {
  vec3 shadow_proj = shadow_pos.xyz / shadow_pos.w;
  shadow_proj = shadow_proj * 0.5 + 0.5;
  if (shadow_proj.z > 1.0) {
    return 0.0;
  }

  float current_depth = shadow_proj.z;
  float bias = 0.07;
  vec2 texel_size = 1.0 / textureSize(depth, 0);

  float shadow = 0.0;
  float visibility = 1.0;
  for(int x = -1; x <= 1; ++x) {
    for(int y = -1; y <= 1; ++y) {
      for (int i = 0; i < 16; i++) {
        int index = int(16.0*random(floor(world_pos.xyz*1000.0), i))%16;
        vec2 perturb = (vec2(x, y)) * texel_size + poisson_disk[index] / 200.0;
        float pcf_depth = texture(depth, vec3(shadow_proj.xy + perturb, (shadow_pos.z - bias)/shadow_pos.w)).r;
        shadow += current_depth - bias > pcf_depth ? 1.0 : 0.0;
      }
    }
  }
  shadow /= 9.0 * 16.0;

  return shadow;
}

