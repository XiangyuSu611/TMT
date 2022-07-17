#version 450 core

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

uniform float u_far;
uniform float u_near;

in vec3 a_position;
out vec3 v_coord;

void main(void) {
    vec4 point_3d;
    point_3d = u_model * vec4(a_position,1.0);
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    v_coord = point_3d.xyz;
}
