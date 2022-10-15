#version 460

// Attributes
layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;

// Uniform variables
uniform mat4 V;
uniform mat4 uProj;
uniform mat4 pose_rot;
uniform mat4 pose_trans;

// Varying variables
out vec3 vColor;

void main() {
    vec4 posWorld = pose_trans * pose_rot * vec4(aPosition, 1);
    vec4 posCam = V * posWorld;

    vColor = aColor;

    gl_Position = uProj * posCam;
}
