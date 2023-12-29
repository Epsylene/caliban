#version 450

layout(binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 proj;
} mvp;

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    // Set the homogenous vertex position (gl_Position) to the
    // current vertex in Vulkan coordinates (inPos) after the
    // MVP transformation (proj*view*model), and output the
    // vertex color (inColor) to the fragment shader
    // (fragColor).
    gl_Position = mvp.proj * mvp.view * mvp.model * vec4(inPos, 0.0, 1.0);
    fragColor = inColor;
}