use crate::{AABB, BVHNode, partitioning::*};
use glam::*;

pub fn build_bvh(
    node_id: usize,
    aabbs: &[AABB],
    centers: &[[f32; 3]],
    indices: &IndicesRef,
    nodes: &mut [BVHNode],
    pool_ptr: &mut usize,
    first_index: usize,
    index_count: usize,
)
{
    if index_count < 3 {
        nodes[node_id].bounds.left_first = first_index as i32;
        nodes[node_id].bounds.count = index_count as i32;
        return;
    }

    nodes[node_id].bounds.left_first = *pool_ptr as i32;
    *pool_ptr += 2;

    let mut split_dim = -1;
    let mut split_cost = std::f32::INFINITY;

    let split_index = partition_sah(aabbs, indices, first_index, index_count, &mut split_dim, &mut split_cost);

    let parent_cost = nodes[node_id].bounds.area() * index_count as f32;
    if split_cost >= parent_cost || split_dim < 0 {
        nodes[node_id].bounds.left_first = first_index as i32;
        nodes[node_id].bounds.count = index_count as i32;
        return;
    }

    let split_dim = split_dim as usize;
    let split_index = split_index as usize;

    let split = centers[indices[(split_dim, split_index)] as usize][split_dim];
    split_indices(centers, indices, first_index, index_count, split_dim, split_index, split);

    nodes[node_id].bounds.count = -1;
    let n_left = split_index - first_index;
    let n_right = first_index + index_count - split_index;

    build_bvh(nodes[node_id].bounds.left_first as usize, aabbs, centers, indices, nodes, pool_ptr, first_index, n_left);
    build_bvh(nodes[node_id].bounds.left_first as usize + 1, aabbs, centers, indices, nodes, pool_ptr, first_index + n_left, n_right);
}

pub fn build_sbvh(
    node_id: usize,
    aabbs: &[AABB],
    triangles: &[impl TriangleStorage],
    centers: &[[f32; 3]],
    indices: &IndicesRef,
    nodes: &mut [BVHNode],
    pool_ptr: &mut usize,
    first_index: usize,
    index_count: usize,
    aabb: &AABB,
    inv_root_surface_area: f32,
) -> i32 {
    nodes[node_id].bounds = *aabb;

    if index_count < 3 {
        nodes[node_id].bounds.left_first = first_index as i32;
        nodes[node_id].bounds.count = index_count as i32;
        return index_count as i32;
    }

    nodes[node_id].bounds.left_first = *pool_ptr as i32;
    *pool_ptr += 2;

    let mut o_split_cost = 0.0;
    let mut o_split_dim = -1;
    let mut o_aabb_left = AABB::new();
    let mut o_aabb_right = AABB::new();
    let o_split_index = partition_object(aabbs, indices, first_index, index_count, &mut o_split_dim, &mut o_split_cost, aabb, &mut o_aabb_left, &mut o_aabb_right);

    assert_ne!(o_split_index, -1);

    let mut s_split_cost = std::f32::INFINITY;
    let mut s_split_dim = -1;
    let mut s_split_plane = 0.0;
    let mut s_aabb_left = AABB::new();
    let mut s_aabb_right = AABB::new();
    let mut s_count_left = 0;
    let mut s_count_right = 0;

    let overlap = AABB::overlap(&o_aabb_left, &o_aabb_right);
    let lambda = if overlap.is_valid() { overlap.area() } else { 0.0 };
    let alpha = 10e-5 as f32;
    let ratio = lambda * inv_root_surface_area;
    assert!(ratio >= 0.0 && ratio <= 1.0);

    if ratio < alpha {
        partition_spatial(triangles, aabbs, indices, first_index, index_count, &mut s_split_dim, &mut s_split_cost, &mut s_split_plane, aabb, &mut s_aabb_left, &mut s_aabb_right, &mut s_count_left, &mut s_count_right);
    }

    let parent_cost = nodes[node_id].bounds.area() * index_count as f32;
    if parent_cost <= o_split_cost && parent_cost <= s_split_cost {
        nodes[node_id].bounds.left_first = first_index as i32;
        nodes[node_id].bounds.count = index_count as i32;
        return index_count as i32;
    }

    // Not a leaf node
    nodes[node_id].bounds.count = -1;

    let mut children_left = indices.with_offset(first_index);
    let mut children_right = [
        vec![0; index_count],
        vec![0; index_count],
        vec![0; index_count],
    ];

    let mut children_left_count = [0; 3];
    let mut children_right_count = [0; 3];

    let mut n_left = 0;
    let mut n_right = 0;

    let mut child_aabb_left = AABB::new();
    let mut child_aabb_right = AABB::new();

    if o_split_cost < s_split_cost {
        let o_split_dim = o_split_dim as usize;
        let split = centers[indices[(o_split_dim as usize, o_split_index as usize)] as usize][o_split_dim];

        for dim in 0..3 {
            for i in first_index..(first_index + index_count) {
                let index = indices[(dim, i)] as usize;
                let mut goes_left = centers[index][o_split_dim] < split;

                if centers[index][o_split_dim] == split {
                    let mut j = o_split_index as usize - 1;
                    while j >= first_index && centers[indices[(o_split_dim, j)] as usize][o_split_dim] == split {
                        if indices[(o_split_dim, j)] as usize == index {
                            goes_left = true;
                            break;
                        }

                        j -= 1;
                    }
                }

                if goes_left {
                    children_left[(dim, children_left_count[dim])] = index as u32;
                    children_left_count[dim] += 1;
                } else {
                    children_right[dim][children_right_count[dim]] = index;
                    children_right_count[dim] += 1;
                }
            }
        }

        // We should have made the same decision (going left/right) in every dimension
        assert!(children_left_count[0] == children_left_count[1] && children_left_count[1] == children_left_count[2]);
        assert!(children_right_count[0] == children_right_count[1] && children_right_count[1] == children_right_count[2]);

        n_left = children_left_count[0];
        n_right = children_right_count[0];

        // Using object split, no duplicates can occur.
        // Thus, left + right should equal the total number of triangles
        assert_eq!(first_index + n_left, o_split_index as usize);
        assert_eq!(n_left + n_right, index_count);

        child_aabb_left = o_aabb_left;
        child_aabb_right = o_aabb_right;
    } else {
        let mut indices_going_left = vec![0; first_index + index_count];
        let mut indices_going_right = vec![0; first_index + index_count];

        let mut rejected_left = 0;
        let mut rejected_right = 0;

        let mut n_1 = s_count_left as f32;
        let mut n_2 = s_count_right as f32;

        let s_split_dim = s_split_dim as usize;

        for i in first_index..(first_index + index_count) {
            let index = indices[(s_split_dim, i)] as usize;
            let (v0, v1, v2) = triangles[index].vertices();
            // let vertices = [v0, v1, v2];

            let mut goes_left =
                v0[s_split_dim as usize] < s_split_plane ||
                    v1[s_split_dim as usize] < s_split_plane ||
                    v2[s_split_dim as usize] < s_split_plane;
            let mut goes_right =
                v0[s_split_dim as usize] >= s_split_plane ||
                    v1[s_split_dim as usize] >= s_split_plane ||
                    v2[s_split_dim as usize] >= s_split_plane;

            assert!(goes_left || goes_right);

            if goes_left && goes_right { // Straddler
                let valid_left = AABB::overlap(&aabbs[index], &s_aabb_left).is_valid();
                let valid_right = AABB::overlap(&aabbs[index], &s_aabb_right).is_valid();

                if valid_left && valid_right {
                    let mut delta_left = s_aabb_left;
                    let mut delta_right = s_aabb_right;

                    delta_left.grow_bb(&aabbs[index]);
                    delta_right.grow_bb(&aabbs[index]);

                    let s_split_left_area = s_aabb_left.area();
                    let s_split_right_area = s_aabb_right.area();

                    // Calculate SAH cost for the 3 different cases
                    let c_split = s_split_left_area * n_1 + s_split_right_area * n_2;
                    let c_1 = delta_left.area() * n_1 + s_split_right_area * (n_2 - 1.0);
                    let c_2 = s_split_left_area * (n_1 - 1.0) + delta_right.area() * n_2;

                    // If C_1 resp. C_2 is cheapest, let the triangle go left resp. right
                    // Otherwise, do nothing and let the triangle go both left and right
                    if c_1 < c_split {
                        if c_2 < c_1 { // C_2 is cheapest, remove from left
                            goes_left = false;
                            rejected_left += 1;

                            n_1 -= 1.0;

                            s_aabb_right.grow_bb(&aabbs[index]);
                        } else { // C_1 is cheapest, remove from right
                            goes_right = false;
                            rejected_right += 1;

                            n_2 -= 1.0;

                            s_aabb_left.grow_bb(&aabbs[index]);
                        }
                    } else if c_2 < c_split { // C_2 is cheapest, remove from left
                        goes_left = false;
                        rejected_left += 1;

                        n_1 -= 1.0;

                        s_aabb_right.grow_bb(&aabbs[index]);
                    }
                }
            }

            // Triangle must go left, right, or both
            assert!(goes_left || goes_right);

            indices_going_left[index] = goes_left as u32;
            indices_going_right[index] = goes_right as u32;
        }

        for dim in 0..3 {
            for i in first_index..(first_index + index_count) {
                let index = indices[(dim, i)] as usize;

                let goes_left = indices_going_left[index] != 0;
                let goes_right = indices_going_right[index] != 0;

                if goes_left {
                    children_left[(dim, children_left_count[dim])] = index as u32;
                    children_left_count[dim] += 1;
                }
                if goes_right {
                    children_right[dim][children_right_count[dim]] = index;
                    children_right_count[dim] += 1;
                }
            }
        }
        // We should have made the same decision (going left/right) in every dimension
        assert!(children_left_count[0] == children_left_count[1] && children_left_count[1] == children_left_count[2]);
        assert!(children_right_count[0] == children_right_count[1] && children_right_count[1] == children_right_count[2]);

        n_left = children_left_count[0];
        n_right = children_right_count[0];

        // The actual number of references going left/right should match the numbers calculated during spatial splitting
        assert_eq!(n_left, s_count_left as usize - rejected_left);
        assert_eq!(n_right, s_count_right as usize - rejected_right);

        // A valid partition contains at least one and strictly less than all
        assert!(n_left > 0 && n_left < index_count);
        assert!(n_right > 0 && n_right < index_count);

        // Make sure no triangles disappeared
        assert!(n_left + n_right >= index_count);

        child_aabb_left = s_aabb_left;
        child_aabb_right = s_aabb_right;
    }

    // Do a depth first traversal, so that we know the amount of indices that were recursively created by the left child
    let number_of_leaves_left = build_sbvh(nodes[node_id].bounds.left_first as usize, aabbs, triangles, centers, indices, nodes, pool_ptr,
                                           first_index, n_left, &child_aabb_left, inv_root_surface_area);
    {
        let indices_x = indices.as_mut_slice(0);
        let indices_y = indices.as_mut_slice(0);
        let indices_z = indices.as_mut_slice(0);

        // Using the depth first offset, we can now copy over the right references
        for i in 0..n_right {
            let index = i + first_index + number_of_leaves_left as usize;
            indices_x[index] = children_right[0][i] as u32;
            indices_y[index] = children_right[1][i] as u32;
            indices_z[index] = children_right[2][i] as u32;
        }
    }

    // Now recurse on the right side
    let number_of_leaves_right = build_sbvh((nodes[node_id].bounds.left_first + 1) as usize, aabbs, triangles, centers, indices, nodes, pool_ptr,
                                            first_index + number_of_leaves_left as usize, n_right, &child_aabb_right, inv_root_surface_area);

    number_of_leaves_left + number_of_leaves_right
}