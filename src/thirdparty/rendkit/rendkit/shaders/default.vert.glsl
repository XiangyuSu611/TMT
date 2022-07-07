#version 450 core
uniform mat4 u_view;
uniform mat4 u_model;
uniform mat4 u_projection;

in vec3 a_position;

#if TPL.use_uvs
in vec2 a_uv;
uniform float u_uv_scale;
#endif

#if TPL.use_normals
in vec3 a_normal;
out vec3 v_normal;
out mat4 v_modelview;
#endif

#if TPL.use_tangents
in vec3 a_tangent;
out vec3 v_tangent;
#endif

#if TPL.use_bitangents
in vec3 a_bitangent;
out vec3 v_bitangent;
#endif

#if TPL.use_radiance_map && TPL.num_shadow_sources > 0
uniform mat4 u_shadow_view[TPL.num_shadow_sources];
uniform mat4 u_shadow_proj[TPL.num_shadow_sources];
out vec4 v_position_shadow[TPL.num_shadow_sources];
#endif

out vec3 v_position;
out vec2 v_uv;

void main() {
    vec4 position = u_model * vec4(a_position, 1.0);
    gl_Position = u_projection * u_view * position;

    v_position = position.xyz;

    #if TPL.use_normals
    v_normal = a_normal;
    v_modelview = u_view * u_model;
    #endif

    #if TPL.use_tangents
    v_tangent = a_tangent;
    #endif

    #if TPL.use_bitangents
    v_bitangent = a_bitangent;
    #endif

    #if TPL.use_radiance_map && TPL.num_shadow_sources > 0
    for (int i = 0; i < TPL.num_shadow_sources; i++) {
      v_position_shadow[i] = u_shadow_proj[i] * u_shadow_view[i] * position;
    }
    #endif

#if TPL.use_uvs
    v_uv  = a_uv * u_uv_scale;
#endif
}
