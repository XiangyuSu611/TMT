#version 450 core

in vec3 v_coord;
out vec4 out_color;

void main(void) {
    out_color = vec4(vec3(v_coord), 1.0);
}
