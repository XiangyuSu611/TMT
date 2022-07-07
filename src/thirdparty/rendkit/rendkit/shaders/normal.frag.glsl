#version 450 core

in vec3 v_normal;
in vec3 v_position;
in mat4 v_modelview;

out vec4 out_color;

void main(void) {
    vec4 normal = inverse(transpose(v_modelview)) * vec4(v_normal, 0);
    out_color = vec4(normalize(normal.xyx), 1.0);
}
