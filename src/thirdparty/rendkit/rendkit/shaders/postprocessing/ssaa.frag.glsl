#version 450 core

uniform sampler2D u_rendtex;
uniform vec2 u_texture_shape;
uniform vec4 u_aa_kernel;

in vec2 v_uv;
out vec4 out_color;


void main() {
    vec2 pos = v_uv.xy;
    vec4 color = vec4(0.0);

    float dx = 1.0 / u_texture_shape.y;
    float dy = 1.0 / u_texture_shape.x;

    // Convolve
    int window_size = 3;
    for (int y = -window_size; y <= window_size; y++) {
        for (int x = -window_size; x <= window_size; x++) {
            float k = u_aa_kernel[int(abs(float(x)))]
            			* u_aa_kernel[int(abs(float(y)))];
            vec2 dpos = vec2(float(x) * dx, float(y) * dy);
            color += texture2D(u_rendtex, pos + dpos) * k;
        }
    }

    // Determine final color
    out_color = color;
}
