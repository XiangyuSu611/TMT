#include "utils/math.glsl"
#include "brdf/fresnel.glsl"


float aittala_ndf_iso(vec3 N, vec3 H, mat2 S, float alpha) {
  mat3 R = mat3(0, 0, N.x,
      0, 0, N.y,
      -N.x, -N.y, 0);

  // Halfway vector in normal-oriented coordinates (so normal is [0,0,1])
  vec3 H_ = H + R * H + 1.0 / (N.z + 1.0) * (R * (R * H));
  vec2 h = H_.xy / H_.z;

  vec2 e = eig(S);
  // Approximate isotropic version from Brady et al.
  float beta = min(e.x, e.y); // The mean eigen value is equal to tr/2.
  float sigma = pow(beta, -1.0/2.0);
  return exp(-pow(sqrt(h.x*h.x+h.y*h.y) / (sigma*sigma), alpha));
}

vec3 aittala_spec(vec3 N, vec3 V, vec3 L, vec3 rho_s, mat2 S, float alpha) {
  vec3 H = normalize(L + V);
  float D = aittala_ndf_iso(N, H, S, alpha);
  float F = fresnel_schlick(F0, V, H) / F0;
  return rho_s * D * F * max(0.0, dot(L, H));
}
