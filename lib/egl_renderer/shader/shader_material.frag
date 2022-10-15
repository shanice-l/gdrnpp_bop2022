#version 460

// Uniform variables
uniform vec3 uLightPosition;  // in world coordinate
uniform vec3 uLightColor; // light color

uniform vec3 uMatAmbient;
uniform vec3 uMatDiffuse;
uniform vec3 uMatSpecular;
uniform float uMatShininess;

// Varying variables
in vec3 Normal;
in vec3 vNormalCam;
in vec3 FragPos;
in vec3 Instance_color;
in vec3 vPosCam;
in vec3 vPos;

// Output variables
layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 InstanceColour;
layout (location = 3) out vec4 PCObject;
layout (location = 4) out vec4 PCColour;


void main() {
    vec3 norm = normalize(Normal);
    vec3 ambient = uMatAmbient * uLightColor;
    vec3 lightDir = normalize(uLightPosition - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * uLightColor * uMatDiffuse;
    vec3 viewDir = normalize(vPosCam - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), uMatShininess);
    vec3 specular = uLightColor * (spec * uMatSpecular);

    outputColour = vec4(ambient + diffuse + specular, 1);
    if(outputColour.x > 1.0) outputColour.x = 1.0;
    if(outputColour.y > 1.0) outputColour.y = 1.0;
    if(outputColour.z > 1.0) outputColour.z = 1.0;

    NormalColour =  vec4(0.5 * vNormalCam + 0.5, 1);
    InstanceColour = vec4(Instance_color, 1);
    PCObject = vec4(vPos, 1);
    PCColour = vec4(vPosCam, 1);
}
