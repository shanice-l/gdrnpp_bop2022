#version 460

uniform mat4 V;
uniform mat4 uProj;

layout (location=0) in vec3 aPosition;
layout (location=1) in vec3 aNormal;
layout (location=2) in vec2 aTexcoord;

void main() {
    gl_Position = uProj * V * vec4(aPosition, 1);
}
