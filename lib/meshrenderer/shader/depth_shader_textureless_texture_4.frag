#version 450 core

#define RENDER_COLOR_ONLY 0
#define RENDER_PHONG 1
#define RENDER_TEXTURE 2

layout (location = 1) uniform float u_light_ambient_w;
layout (location = 2) uniform float u_light_diffuse_w;
layout (location = 3) uniform float u_light_specular_w;
// 0: color only; 1: phong; 2: texture
layout (location = 4) uniform int u_render_type;

layout (location = 5) uniform sampler2D u_texture;

// uniform vec3 light_position;  // in world coordinate
// uniform vec3 light_color; // light color

// Varying variables
in vec2 v_texcoord;

in vec3 v_color;
in vec3 v_view;
in vec3 ViewDir;

in vec3 v_L;       // Vector to the light
in vec3 v_normal;  // Normal in eye coords.
// in vec3 v_P;       // gl_Position.xyz;
// in vec3 vPosCam;

// Output variables
layout (location = 0) out vec4 rgb;
layout (location = 1) out vec4 depth;
layout (location = 2) out vec4 rgb_normals;


void main(void) {

    vec3 Normal = normalize(v_normal);
    if (u_render_type == RENDER_COLOR_ONLY)
    { // color only
        rgb = vec4(v_color, 1.0);
    }
    else if (u_render_type == RENDER_PHONG)
    {  // phong vertex color
        vec3 LightDir = normalize(v_L);
        vec3 ViewDir = normalize(v_view);

        vec3 diffuse = max(dot(Normal, LightDir), 0.0) * v_color;
        vec3 R = reflect(-LightDir, Normal);
        vec3 specular = max(dot(R, ViewDir), 0.0) * v_color;

        rgb = vec4(u_light_ambient_w  * v_color +
                u_light_diffuse_w  * diffuse +
                u_light_specular_w * specular, 1.0);
    }
    else
    {  // textured
        vec3 color = texture2D(u_texture, v_texcoord).xyz;
        vec3 LightDir = normalize(v_L);
        vec3 ViewDir = normalize(v_view);

        vec3 diffuse = max(dot(Normal, LightDir), 0.0) * color;
        vec3 R = reflect(-LightDir, Normal);
        vec3 specular = max(dot(R, ViewDir), 0.0) * color;

        rgb = vec4(u_light_ambient_w  * color +
                u_light_diffuse_w  * diffuse +
                u_light_specular_w * specular, 1.0);
    }

    if (rgb.x > 1.0) rgb.x = 1.0;
    if (rgb.y > 1.0) rgb.y = 1.0;
    if (rgb.z > 1.0) rgb.z = 1.0;

    rgb_normals = vec4(Normal * 0.5 + 0.5, 1.0); // transforms from [-1,1] to [0,1]
    depth = vec4(v_view.z, 0.0, 0.0, 1.0);
}
