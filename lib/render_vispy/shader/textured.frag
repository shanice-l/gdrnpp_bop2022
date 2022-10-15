uniform float u_ambient;
uniform float u_specular;
uniform float u_shininess;
uniform vec3 u_light_dir;
uniform vec3 u_light_col;

uniform sampler2D u_tex;

varying vec2 v_texcoord;
varying vec3 v_V;
varying vec3 v_P;

void main() {
    vec3 N = normalize(cross(dFdy(v_P), dFdx(v_P))); // N is the world normal
    vec3 V = normalize(v_V);
    vec3 R = reflect(V, N);
    vec3 L = normalize(u_light_dir);

    vec3 color = texture2D(u_tex, v_texcoord).xyz;
    vec3 ambient = color * u_light_col * u_ambient;
    vec3 diffuse = color * u_light_col * max(dot(L, N), 0.0);
    float specular = u_specular * pow(max(dot(R, L), 0.0), u_shininess);
    gl_FragColor = vec4(ambient + diffuse + specular, 1.0);

}
