#version 450

layout(location = 0) out vec3 fragColor;

// Positions of the triangle vertices
vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

// Colors of the triangle vertices
vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() {
    // Set the homogenous vertex position (gl_Position) for the
    // current vertex (of index 'gl_VertexIndex'), and output
    // the vertex color to the fragment shader.
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}