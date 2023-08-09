#version 450

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    // Set the homogenous vertex position (gl_Position) for the
    // current vertex (inPos), and output the vertex color
    // (inColor) to the fragment shader (fragColor).
    gl_Position = vec4(inPos, 0.0, 1.0);
    fragColor = inColor;
}