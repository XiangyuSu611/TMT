const float F0 = 0.04;

float fresnel_schlick(float F0, vec3 V, vec3 H) {
  float VdotH = max(0, dot(V, H));
  float F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
  return max(F, 0);
}
