#version 460

// Attributes
layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aNormal;

// Uniform variables
uniform mat4 V;
uniform mat4 uProj;
uniform mat4 pose_rot;
uniform mat4 pose_trans;
uniform vec3 instance_color;

// Varying variables
out vec3 vPos;
out vec3 vPosCam;
out vec3 Normal;
out vec3 vNormalCam;
out vec3 FragPos;

out vec3 Instance_color;



void main() {
    vec4 posWorld = pose_trans * pose_rot * vec4(aPosition, 1);
    vec4 posCam = V * posWorld;

    vPos = aPosition;
    vPosCam = posCam.xyz / posCam.w;
    FragPos = vec3(posWorld.xyz / posWorld.w); // in world coordinate
    Normal = normalize(mat3(pose_rot) * aNormal); // in world coordinate
    // Normal in camera coordinate
    vNormalCam = normalize(mat3(V) * mat3(pose_rot) * aNormal);

    Instance_color = instance_color;

    gl_Position = uProj * posCam;
}
