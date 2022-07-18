#version 450 core

in vec3 v_bitangent;
in vec3 v_position;
out vec4 out_color;

void main(void) {
  if (length(v_bitangent) < 0.1) {
    out_color = vec4(0.0, 0.0, 0.0, 1.0);
  } else {
    out_color = vec4(normalize(v_bitangent), 1.0);
  }
}
