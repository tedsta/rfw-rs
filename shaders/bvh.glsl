struct BVHNode {
    float min_x;
    float min_y;
    float min_z;
    int left_first;

    float max_x;
    float max_y;
    float max_z;
    int count;
};

bool intersect_bvh_node(vec3 origin, vec3 dir_inverse, BVHNode node, inout float t_min, float t) {
    float t1 = (node.min_x - origin.x) * dir_inverse.x;
    float t2 = (node.max_x - origin.x) * dir_inverse.x;

    t_min = min(t1, t2);
    float t_max = max(t1, t2);

    t1 = (node.min_y - origin.y) * dir_inverse.y;
    t2 = (node.max_y - origin.y) * dir_inverse.y;

    t_min = max(t_min, min(t1, t2));
    t_max = min(t_max, max(t1, t2));

    t1 = (node.min_z - origin.z) * dir_inverse.z;
    t2 = (node.max_z - origin.z) * dir_inverse.z;

    t_min = max(t_min, min(t1, t2));
    t_max = min(t_max, max(t1, t2));

    return t_max > t_min && t_min < t;
} 