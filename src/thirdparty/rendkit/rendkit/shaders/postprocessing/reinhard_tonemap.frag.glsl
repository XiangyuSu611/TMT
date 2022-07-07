#version 450 core

uniform sampler2D u_rendtex;
uniform float u_thres;

in vec2 v_uv;
out vec4 out_color;

void main() {
    vec4 L = texture2D(u_rendtex, v_uv);
    float thres2 = u_thres * u_thres;
    L.rgb = (L.rgb * (1 + L.rgb / thres2)) / (vec3(1.0) + L.rgb);
    out_color = L;
}
