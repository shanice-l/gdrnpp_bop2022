#version 460

// Uniform variables
uniform sampler2D uTexture;

// Varying variables
in vec2 vTexcoord;

// Output variables
layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 InstanceColour;
layout (location = 3) out vec4 PCObject;
layout (location = 4) out vec4 PCColour;


void main() {

    outputColour =  texture2D(uTexture, theCoords);

    if(outputColour.x > 1.0) outputColour.x = 1.0;
    if(outputColour.y > 1.0) outputColour.y = 1.0;
    if(outputColour.z > 1.0) outputColour.z = 1.0;

    NormalColour =  vec4(0,0,0,0);
    InstanceColour = vec4(0,0,0,0);
    PCObject = vec4(0,0,0,0);
    PCColour = vec4(0,0,0,0);
}
