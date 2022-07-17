#version 450 core

uniform sampler2D u_rendtex;
uniform float u_exposure;

in vec2 v_uv;
out vec4 out_color;

void main() {
    vec4 color = texture2D(u_rendtex, v_uv);
    color.rgb = vec3(1.0) - exp(-color.rgb * u_exposure);
    out_color = color;
}
