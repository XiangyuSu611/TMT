#version 450 core

uniform sampler2D u_texture;

in vec2 v_uv;
out vec4 out_color;

void main(void) {
  out_color = texture(u_texture, v_uv);
}
