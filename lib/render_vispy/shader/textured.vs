uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

attribute vec3 a_position;
attribute vec2 a_texcoord;

varying vec2 v_texcoord;
varying vec3 v_V;
varying vec3 v_P;

void main() {
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
    v_texcoord = a_texcoord;
    v_P = gl_Position.xyz; // v_P is the world position
    v_V = (u_view * u_model * vec4(a_position, 1.0)).xyz;
}
