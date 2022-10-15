#version 450 core

in vec3 v_color;
in vec3 v_view;
in vec3 ViewDir;

in vec3 v_L;
in vec3 v_normal;

layout (location = 1) uniform float uLightAmbientWeight;
layout (location = 2) uniform float u_light_diffuse_w;
layout (location = 3) uniform float u_light_specular_w;
layout (location = 4) uniform int u_color_only;

layout (location = 0) out vec3 rgb;
layout (location = 1) out float depth;
layout (location = 2) out vec3 rgb_normals;

void main(void) {

    vec3 Normal = normalize(v_normal);
    if (bool(u_color_only) == false) {
        vec3 LightDir = normalize(v_L);
        vec3 ViewDir = normalize(v_view);

        vec3 diffuse = max(dot(Normal, LightDir), 0.0) * v_color;
        vec3 R = reflect(-LightDir, Normal);
        vec3 specular = max(dot(R, ViewDir), 0.0) * v_color;


        rgb = vec3(uLightAmbientWeight  * v_color +
                u_light_diffuse_w  * diffuse +
                u_light_specular_w * specular);
    }
    else {  // color only
        rgb = v_color;
    }

    if(rgb.x > 1.0) rgb.x = 1.0;
    if(rgb.y > 1.0) rgb.y = 1.0;
    if(rgb.z > 1.0) rgb.z = 1.0;

    rgb_normals = vec3(Normal * 0.5 + 0.5); // transforms from [-1,1] to [0,1]
    depth = v_view.z;
}
