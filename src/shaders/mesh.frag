#version 450
#extension GL_GOOGLE_include_directive : require

void createTangentSpace(const vec3 N, inout vec3 T, inout vec3 B);
vec3 tangentToWorld(const vec3 s, const vec3 N, const vec3 T, const vec3 B);
vec3 worldToTangent(const vec3 s, const vec3 N, const vec3 T, const vec3 B);

#include "tools.glsl"

layout(location = 0) out vec4 Color;

layout(location = 0) in vec4 V;
layout(location = 1) in vec3 N;
layout(location = 2) in flat uint MID;
layout(location = 3) in vec2 TUV;

struct Material {
    vec3 color;
    vec3 specular;
    float opacity;
    float roughness;
    int diffuse_tex;
    int normal_tex;
};

layout(set = 0, binding = 1) buffer readonly Materials {
    Material materials[];
};

layout(set = 0, binding = 2) uniform sampler Sampler;

layout(set = 2, binding = 0) uniform texture2D AlbedoT;
layout(set = 2, binding = 1) uniform texture2D NormalT;


void main() {
    vec3 color = materials[MID].color;
    vec3 normal = N;

    if (materials[MID].diffuse_tex > 0) {
        vec4 t_color = texture(sampler2D(AlbedoT, Sampler), TUV).rgba;
        if (t_color.a < 0.5) {
            discard;
        }

        color = t_color.xyz;
    }
    if (materials[MID].normal_tex > 0) {
        normal = texture(sampler2D(NormalT, Sampler), TUV).rgb;
        vec3 T, B;
        createTangentSpace(N, T, B);
        normal = tangentToWorld(normal, N, T, B);
    }

    Color = vec4(color, V.w);
}