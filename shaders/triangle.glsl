#define T_EPSILON 1e-6

bool intersect_triangle(vec3 origin, vec3 direction, float t_min, inout float t, vec3 p0, vec3 p1, vec3 p2) {
    vec3 e1 = p1 - p0;
    vec3 e2 = p2 - p0;

    vec3 h = cross(direction, e2);
    float a = dot(e1, h);
    if (a > -T_EPSILON && a < T_EPSILON) {
        return false;
    }

    float f = 1.0 / a;
    vec3 s = origin - p0;
    float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return false;
    }

    vec3 q = cross(s, e1);
    float v = f * dot(direction, q);
    if (v < 0.0 || (u + v) > 1.0) {
        return false;
    }

    float new_t = f * dot(e2, q);

    if (new_t > t_min && new_t < t) {
        t = new_t;
        return true;
    }

    return false;
}