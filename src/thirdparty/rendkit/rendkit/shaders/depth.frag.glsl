#version 450 core

in float v_depth;
out vec4 out_color;

void main(void) {
    out_color = vec4(vec3(v_depth), 1.0);
}
