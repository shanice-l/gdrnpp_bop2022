#version 460

layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 InstanceColour;
layout (location = 3) out vec4 PCColour;

void main() {
    outputColour = vec4(0.1, 0.1, 0.1, 1.0);
    NormalColour = vec4(0,0,0,0);
    InstanceColour = vec4(0,0,0,0);
    PCColour = vec4(0,0,0,0);

}
