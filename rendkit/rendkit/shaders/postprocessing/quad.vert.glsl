#version 450 core

in vec2 a_position;
in vec2 a_uv;
out vec2 v_uv;

void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_uv = a_uv;
}
