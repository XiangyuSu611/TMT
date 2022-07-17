#version 450 core
uniform sampler2D input_tex;
uniform sampler2D input_depth;
uniform vec3 u_cam_pos;

in vec3 v_position;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;
in vec2 v_uv;
in vec3 v_pos_clip_space;
in float v_depth;

out vec4 out_color;


void main() {
	bool normal_faces_camera = dot(v_normal, u_cam_pos - v_position) > 0.0;
	vec2 pos_as_uv = (v_pos_clip_space.xy + 1.0) / 2.0;
	pos_as_uv.y = 1.0 - pos_as_uv.y;
	float input_depth = texture2D(input_depth, pos_as_uv).x;
	if (normal_faces_camera && abs(input_depth - v_depth) < 20.0/255) {
		out_color = texture2D(input_tex, pos_as_uv);
	} else {
		out_color = vec4(1.0, 0.0, 1.0, 0.0);
	}
}
