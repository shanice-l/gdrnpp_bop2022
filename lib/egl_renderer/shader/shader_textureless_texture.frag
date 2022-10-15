#version 460

// Uniform variables
uniform sampler2D uTexture;

uniform vec3 uLightPosition;  // in world coordinate
uniform vec3 uLightColor; // light color
uniform int uUseTexture;  // use texture

uniform vec3 uMatAmbient;
uniform vec3 uMatDiffuse;
uniform vec3 uMatSpecular;
uniform float uMatShininess;

uniform float uLightAmbientWeight;
uniform float uLightDiffuseWeight;
uniform float uLightSpecularWeight;


// Varying variables
in vec2 vTexcoord;
in vec3 vColor;
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
    vec3 lightDir = normalize(uLightPosition - FragPos);
    float diff = max(dot(Normal, lightDir), 0.0);

    if (bool(uUseTexture)) {
        vec3 norm = normalize(Normal);
        vec3 ambient = uMatAmbient * uLightColor;
        vec3 diffuse = diff * uLightColor * uMatDiffuse;
        vec3 viewDir = normalize(vPosCam - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), uMatShininess);
        vec3 specular = uLightColor * (spec * uMatSpecular);
        outputColour =  texture(uTexture, vTexcoord) * vec4(diffuse + ambient + specular, 1);
    }
    else {
        int aae_phong = 0;
        if (false == bool(aae_phong)) {
            float ambientStrength = 0.2;
            vec3 ambient = ambientStrength * uLightColor;
            vec3 diffuse = diff * uLightColor;
            outputColour = vec4(vColor, 1) * vec4(diffuse + ambient, 1);
        }

        if (bool(aae_phong)) {  // TODO: figure out how to make it work
            vec3 diffuse = max(dot(vNormalCam, lightDir), 0.0) * vColor;
            // vec3 R = reflect(-lightDir, vNormalCam);

            vec3 norm = normalize(Normal);
            vec3 reflectDir = reflect(-lightDir, norm);
            //vec3 specular = max(dot(R, viewDir), 0.0) * vColor;
            vec3 viewDir = normalize(vPosCam);  // - FragPos);  // check viewDir

            vec3 specular = max(dot(reflectDir, viewDir), 0.0) * vColor;

            outputColour = vec4(uLightAmbientWeight  * vColor +
                                uLightDiffuseWeight  * diffuse +
                                uLightSpecularWeight * specular, 1.0);
        }
    }

    if(outputColour.x > 1.0) outputColour.x = 1.0;
    if(outputColour.y > 1.0) outputColour.y = 1.0;
    if(outputColour.z > 1.0) outputColour.z = 1.0;

    NormalColour =  vec4(0.5 * vNormalCam + 0.5, 1);
    InstanceColour = vec4(Instance_color, 1);
    PCObject = vec4(vPos, 1);
    PCColour = vec4(vPosCam, 1);
}
