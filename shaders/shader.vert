#version 450

layout(binding = 0) uniform MVP {
    mat4 view;
    mat4 proj;
} vp;

layout(push_constant) uniform PushConstants {
    mat4 model;
} pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    // Set the homogenous vertex position (gl_Position) to the
    // current vertex in Vulkan coordinates (inPos) after the
    // MVP transformation (proj*view*model), and output the
    // vertex color (inColor) to the fragment shader
    // (fragColor), as well as the texture coordinate
    // (inTexCoord) to the fragment shader (fragTexCoord).
    gl_Position = vp.proj * vp.view * pc.model * vec4(inPos, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}