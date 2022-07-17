#version 450 core
#include "utils/math.glsl"
#include "utils/sampling.glsl"
#include "brdf/aittala.glsl"
#include "envmap/dual_paraboloid.glsl"

#define LIGHT_POINT 0
#define LIGHT_DIRECTIONAL 1
#define LIGHT_AMBIENT 2

uniform sampler2D u_diff_map;
uniform sampler2D u_spec_map;
uniform sampler2D u_spec_shape_map;
uniform sampler2D u_normal_map;
uniform float u_alpha;

uniform vec2 u_sigma_range;
uniform sampler2D u_cdf_sampler;
uniform sampler2D u_pdf_sampler; // Normalization factor for PDF.
uniform vec3 u_cam_pos;

#if TPL.change_color
#include "utils/colors.glsl"
uniform vec3 u_mean_old;
uniform vec3 u_mean_new;
uniform vec3 u_std_old;
uniform vec3 u_std_new;
#endif

#if TPL.num_lights > 0
uniform float u_light_intensity[TPL.num_lights];
uniform vec3 u_light_position[TPL.num_lights];
uniform vec3 u_light_color[TPL.num_lights];
uniform int u_light_type[TPL.num_lights];
#endif

#if TPL.use_radiance_map
uniform samplerCube u_irradiance_map;
uniform sampler2D u_radiance_upper;
uniform sampler2D u_radiance_lower;
uniform float u_radiance_scale;
uniform vec2 u_cubemap_size;
#endif

#if TPL.num_shadow_sources > 0
#include "utils/shadow.glsl"
uniform sampler2DShadow u_shadow_depth[TPL.num_shadow_sources];
in vec4 v_position_shadow[TPL.num_shadow_sources];
#endif

const float NUM_LIGHTS = TPL.num_lights;

in vec3 v_position;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;
in vec2 v_uv;

out vec4 out_color;


vec3 compute_irradiance(vec3 N, vec3 L, vec3 light_color) {
  float cosine_term = max(.0, dot(N, L));
  return cosine_term * max(vec3(0.0), light_color);
}

vec2 compute_sample_angles(float sigma, vec2 xi) {
  float phi = 2.0f * M_PI * xi.x;
  float sigma_samp = (sigma - u_sigma_range.x) / (u_sigma_range.y - u_sigma_range.x);
  float theta = texture2D(u_cdf_sampler, vec2(xi.y, sigma_samp)).r;
  return vec2(phi, theta);
}

float get_pdf_value(float sigma, vec2 xi) {
  float sigma_samp = (sigma - u_sigma_range.x) / (u_sigma_range.y - u_sigma_range.x);
  return texture(u_pdf_sampler, vec2(xi.y, sigma_samp)).r;
}


void main() {
  vec3 V = normalize(u_cam_pos - v_position);

  float lod = textureQueryLod(u_diff_map, v_uv).x;
  vec3 rho_d = textureLod(u_diff_map, v_uv, lod).rgb;
  #if TPL.change_color
  rho_d = rgb2lab(rho_d) - u_mean_old;
  rho_d = rho_d / u_std_old * u_std_new + u_mean_new;
  rho_d = lab2rgb(rho_d);
  #endif
  vec3 rho_s = textureLod(u_spec_map, v_uv, lod).rgb;
  vec3 specv = textureLod(u_spec_shape_map, v_uv, lod).rgb;

  mat3 TBN = mat3(v_tangent, v_bitangent, v_normal);
  vec3 N = normalize(TBN * textureLod(u_normal_map, v_uv, lod).rgb);

  // Flip normal if back facing.
//  bool is_back_facing = dot(V, v_normal) < 0;
//  if (is_back_facing) {
//    N *= -1;
//  }

  float shadowness = 0.0;
	#if TPL.num_shadow_sources > 0
	for (int i = 0; i < TPL.num_shadow_sources; i++) {
    shadowness += compute_shadow(v_position, v_position_shadow[i], u_shadow_depth[i]);
	}
  shadowness /= TPL.num_shadow_sources * 2.0;
	#endif

  mat2 S = mat2(specv.x, specv.z,
      specv.z, specv.y);

  vec3 total_radiance = vec3(0.0);

  #if TPL.use_radiance_map
  total_radiance += rho_d * texture(u_irradiance_map, N).rgb;

  vec3 specular = vec3(0);
  vec2 e = eig(S);
  float sigma = pow(min(e.x, e.y), -1.0/4.0);
  uint N_SAMPLES = 200u;
  for (uint i = 0u; i < N_SAMPLES; i++) {
    vec2 xi = hammersley(i, N_SAMPLES); // Use psuedo-random point set.
    vec2 sample_angle = compute_sample_angles(sigma, xi);
    float phi = sample_angle.x;
    float theta = sample_angle.y;
    vec3 H = angle_to_vec(phi, theta);
    H = local_to_world(H, N);
    vec3 L = reflect(-V, H);
    float pdf = get_pdf_value(sigma, xi);
		float lod = compute_lod(pdf, N_SAMPLES, u_cubemap_size.x, u_cubemap_size.y);
    vec3 light_color;
    vec2 dp_uv = dualp_world_to_tex(L, 1.2);
    if (L.y > 0) {
      light_color = textureLod(u_radiance_upper, dp_uv, lod).rgb;
    } else {
      light_color = textureLod(u_radiance_lower, dp_uv, lod).rgb;
    }
    specular += compute_irradiance(N, L, light_color)
        * aittala_spec_is(N, V, L, rho_s, S, u_alpha, pdf) / float(N_SAMPLES);
  }
  total_radiance += specular;
  total_radiance *= u_radiance_scale;
  #endif

  #if TPL.num_lights > 0
  for (int i = 0; i < NUM_LIGHTS; i++) {
    vec3 irradiance = vec3(0);
    if (u_light_type[i] == LIGHT_AMBIENT) {
      irradiance = u_light_intensity[i] * u_light_color[i];
    } else {
      vec3 L;
      float attenuation = 1.0;
      if (u_light_type[i] == LIGHT_POINT) {
        L = u_light_position[i] - v_position;
        attenuation = 1.0 / dot(L, L);
        L = normalize(L);
      } else if (u_light_type[i] == LIGHT_DIRECTIONAL) {
        L = normalize(u_light_position[i]);
      } else {
        continue;
      }
      bool is_light_visible = dot(L, N) >= 0;
      if (is_light_visible) {
        irradiance = compute_irradiance(N, L, u_light_intensity[i] * u_light_color[i]);
        total_radiance += aittala_spec(N, V, L, rho_s, S, u_alpha) * irradiance;
      }
    }
    total_radiance += rho_d * irradiance;
  }
  #endif

	total_radiance *= (1.0 - shadowness);
  out_color = vec4(max(vec3(.0), total_radiance), 1.0);    // rough gamma
}
