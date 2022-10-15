#version 450 core

// Attributes
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 color;
layout (location = 3) in vec2 a_texcoord;

// Uniform variables
layout (binding = 0) readonly buffer SCENE_BUFFER {
    mat4 view;
    mat4 projection;
    vec3 viewPos;
};

layout (location = 0) uniform vec3 u_light_eye_pos;

// Varying variables
out vec2 v_texcoord;
out vec3 v_color;
out vec3 v_view;

out vec3 v_L;
out vec3 v_normal;
// out vec3 v_P;  //gl_Position.xyz
// out v_eye_pos;

void main(void) {
    // assume model matrix is I
    vec4 P = view * vec4(position, 1.0);  // position in camera space
    v_view = -P.xyz;
    gl_Position = projection * P;  // projection * view * position
    // v_P = gl_Position.xyz;
    v_color = color;
    v_texcoord = a_texcoord;

    mat4 u_nm = transpose(inverse(view));  // normal matrix

    // The following points/vectors are expressed in the eye (camera) coordinates.
    vec3 v_eye_pos = P.xyz; // Vertex position in eye coords.
    v_L = normalize(u_light_eye_pos - v_eye_pos); // Vector to the light
    v_normal = normalize(u_nm * vec4(normal, 1.0)).xyz; // Normal in eye coords.
}
