#version 450

#include "lights.glsl"

layout(location = 0) in vec4 Vertex;
layout(location = 1) in uvec4 Joints;
layout(location = 2) in vec4 Weights;

layout(set = 0, binding = 0) uniform Locals {
    LightInfo info;
};

layout(set = 1, binding = 0) uniform I {
    mat4 Transform;
    mat4 InverseTransform;
};

layout(set = 2, binding = 0) buffer readonly SkinMatrices { mat4 jointMatrices[]; };

layout (location = 0) out vec4 LightSpaceV;
layout (location = 1) out vec4 V;

void main() {
    const mat4 skinMatrix = (Weights.x * jointMatrices[Joints.x]) + (Weights.y * jointMatrices[Joints.y]) + (Weights.z * jointMatrices[Joints.z]) + (Weights.w * jointMatrices[Joints.w]);
    const vec4 v = Transform * skinMatrix * Vertex;
    V = v;

    const vec4 Light_V = info.MP * vec4(v.xyz, 1.0);
    LightSpaceV = Light_V;
    gl_Position = Light_V;
}