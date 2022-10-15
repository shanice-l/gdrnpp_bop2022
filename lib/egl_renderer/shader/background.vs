#version 460

layout (location=0) in vec2 aPosition;
layout (location=2) in vec2 aTexcoord;
out vec2 theCoords;

void main() {
    gl_Position = vec4(aPosition, 0, 1.0);
    theCoords = aTexcoord;
}
