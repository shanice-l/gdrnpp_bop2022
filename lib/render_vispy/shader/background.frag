uniform sampler2D u_tex;

varying vec2 v_texcoord;

void main() {
    gl_FragColor = texture2D(u_tex, v_texcoord);
}
