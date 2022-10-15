varying vec3 v_color;
varying vec3 v_V;
varying vec3 v_P;

void main() {
    gl_FragColor = vec4(v_color, 1.0);
}
