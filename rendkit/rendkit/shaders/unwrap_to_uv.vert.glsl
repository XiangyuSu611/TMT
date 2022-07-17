#version 450 core
uniform mat4 u_view;
uniform mat4 u_model;
uniform mat4 u_projection;
uniform float u_near;
uniform float u_far;

in vec3 a_position;
in vec2 a_uv;
in vec3 a_normal;
in vec3 a_tangent;
in vec3 a_bitangent;

out vec3 v_position;
out vec3 v_normal;
out vec3 v_tangent;
out vec3 v_bitangent;
out vec2 v_uv;
out vec3 v_pos_clip_space;
out float v_depth;


void main() {
    vec4 pos_clip_space = u_projection * u_view * u_model * vec4(a_position, 1.0);
    vec2 uv_ndc = a_uv * 2 - 1.0;
    gl_Position = vec4(uv_ndc, 0.0, 1.0);

    vec4 point_3d = u_view * u_model * vec4(a_position,1.0);
    v_depth = (point_3d.z - u_far) / (u_near - u_far);
    v_position = a_position.xyz;
    v_normal = a_normal;
    v_tangent = a_tangent;
    v_bitangent = a_bitangent;
    v_uv  = a_uv;
    v_pos_clip_space = pos_clip_space.xyz / pos_clip_space.w;
}
