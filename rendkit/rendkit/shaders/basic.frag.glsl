#version 450 core

uniform vec3 u_color;

out vec4 out_color;

void main(void) {
    out_color = vec4(vec3(u_color), 1.0);
}
