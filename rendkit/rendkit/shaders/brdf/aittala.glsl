#include "brdf/fresnel.glsl"


float aittala_ndf(vec3 N, vec3 H, mat2 S, float alpha) {
  mat3 R = mat3(0, 0, N.x,
      0, 0, N.y,
      -N.x, -N.y, 0);

  // Halfway vector in normal-oriented coordinates (so normal is [0,0,1])
  vec3 H_ = H + R * H + 1.0 / (N.z + 1.0) * (R * (R * H));
  vec2 h = H_.xy / H_.z;

  // Aittala (2015) NDF.
  vec2 hT_S = h * S;
  float hT_S_h = dot(hT_S, h); // h^T S h
  return exp(-pow(hT_S_h, alpha/2));
}

vec3 aittala_spec(vec3 N, vec3 V, vec3 L, vec3 rho_s, mat2 S, float alpha) {
  vec3 H = normalize(L + V);
  float D = aittala_ndf(N, H, S, alpha);
  float F = fresnel_schlick(F0, V, H) / F0;
  return rho_s * D * F * max(0.0, dot(L, H));
}


vec3 aittala_spec_is(vec3 N, vec3 V, vec3 L, vec3 rho_s, mat2 S, float alpha, float pdf) {
  vec3 H = normalize(L + V);
  // Avoid a divide by zero by making really small probabilities just black.
  // This makes sense to anyway since the PDF is proportional to the NDF.
  if (pdf < 0.0001) {
    return vec3(0);
  }
  float F = fresnel_schlick(F0, V, H) / F0;
  float D = aittala_ndf(N, H, S, alpha);
  return rho_s * F * D * max(0.0, dot(L, H)) / pdf;
}

