use crate::aabb::AABB;
use glam::Vec3;

pub fn calculate_bounds(aabbs: &[AABB], indices: &[u32], range: (usize, usize)) -> AABB {
    let mut new_aabb = AABB::new();

    for i in range.0..range.1 {
        new_aabb.grow_bb(&aabbs[indices[i] as usize]);
    }

    new_aabb.ensure_non_zero();

    assert!(new_aabb.is_valid());

    new_aabb
}

pub fn split_indices(
    centers: &[Vec3],
    indices: [&mut [u32]; 3],
    first_index: usize,
    index_count: usize,
    split_dim: usize,
    split_id: usize,
    split: f32,
) {
    let mut temp = vec![0; indices[0].len()];

    for dim in 0..3 {
        if dim == split_dim {
            continue;
        }

        let mut left = 0;
        let mut right = split_id - first_index;

        for i in first_index..index_count {
            let mut goes_left = centers[indices[dim][i] as usize][split_dim] < split;

            if centers[indices[dim][i] as usize][split_dim] == split {
                // In case the current primitive has the same coordinate as the one we split on along the split dimension,
                // We don't know whether the primitive should go left or right.
                // In this case check all primitive indices on the left side of the split that
                // have the same split coordinate for equality with the current primitive index i

                let mut j = split_id - 1;

                while j >= first_index && centers[indices[split_dim][j] as usize][split_dim] == split {
                    if indices[split_dim][j] == indices[dim][i] {
                        goes_left = true;
                        break;
                    }

                    j -= 1;
                }
            }

            if goes_left {
                temp[left] = indices[dim][i];
                left += 1;
            } else {
                temp[right] = indices[dim][i];
                right -= 1;
            }
        }

        // If these conditions are not met the copy below is invalid
        assert_eq!(left, split_id - first_index);
        assert_eq!(right, index_count);

        unsafe {
            let dest = indices[dim].as_mut_ptr().add(first_index);
            let source = temp.as_ptr();
            std::ptr::copy(source, dest, index_count);
        }
    }
}

pub fn partition_sah(
    aabbs: &[AABB],
    indices: [&mut [u32]; 3],
    first_index: usize,
    index_count: usize,
    split_dim: &mut i32,
    split_cost: &mut f32,
) -> i32 {
    let mut min_split_cost = std::f32::INFINITY;
    let mut min_split_index = -1;
    let mut min_split_dimension = -1;
    let mut sah = vec![std::f32::INFINITY; index_count];

    for dim in 0..3 {
        let mut aabb_left = AABB::new();
        let mut aabb_right = AABB::new();

        for i in 0..index_count {
            let index = first_index + i;
            aabb_left.grow_bb(&aabbs[indices[0][index] as usize]);
            sah[i] = aabb_left.area() * (i + 1) as f32;
        }

        for i in (1..index_count).rev() {
            let index = first_index + i;
            aabb_right.grow_bb(&aabbs[indices[dim][index] as usize]);
            let cost = sah[i - 1] + aabb_right.area() * (index_count - 1) as f32;

            if cost < min_split_cost {
                min_split_cost = cost;
                min_split_index = (first_index + i) as i32;
                min_split_dimension = dim as i32;
            }
        }
    }

    *split_cost = min_split_cost;
    *split_dim = min_split_dimension;
    min_split_index
}


pub fn partition_object(
    aabbs: &[AABB],
    indices: [&mut [u32]; 3],
    first_index: usize,
    index_count: usize,
    split_dim: &mut i32,
    split_cost: &mut f32,
    node_aabb: &AABB,
    aabb_left: &mut AABB,
    aabb_right: &mut AABB,
) -> i32 {
    let mut min_split_cost = std::f32::INFINITY;
    let mut min_split_index = -1;
    let mut min_split_dimension = -1;
    let mut sah = vec![std::f32::INFINITY; index_count];

    let mut bounds_left = vec![AABB::new(); index_count];
    let mut bounds_right = vec![AABB::new(); index_count + 1];

    for dim in 0..3 {
        bounds_left[0] = AABB::new();
        bounds_right[index_count] = AABB::new();

        for i in 1..index_count {
            bounds_left[i] = bounds_left[i - 1];
            bounds_left[i].grow_bb(&aabbs[indices[dim][first_index + i - 1] as usize]);
            bounds_left[i] = AABB::overlap(&bounds_left[i], node_aabb);

            sah[i] = bounds_left[i].area() * i as f32;
        }

        for i in (1..index_count).rev() {
            bounds_right[i] = bounds_right[i + 1];
            bounds_right[i].grow_bb(&aabbs[indices[dim][first_index + i] as usize]);
            bounds_right[i] = AABB::overlap(&bounds_right[i], node_aabb);

            let cost = sah[i] + bounds_right[i].area() * (index_count - i) as f32;

            if cost < min_split_cost {
                min_split_cost = cost;
                min_split_index = (first_index + i) as i32;
                min_split_dimension = dim as i32;

                assert!(!bounds_left[i].is_empty());
                assert!(!bounds_right[i].is_empty());

                *aabb_left = bounds_left[i];
                *aabb_right = bounds_right[i];
            }
        }
    }

    *split_cost = min_split_cost;
    *split_dim = min_split_dimension;
    min_split_index
}

#[derive(Debug, Copy, Clone)]
pub struct Bin {
    aabb: AABB,
    entries: u32,
    exits: u32,
}

pub trait TriangleStorage {
    fn vertex0(&self) -> Vec3;
    fn vertex1(&self) -> Vec3;
    fn vertex2(&self) -> Vec3;
    fn vertices(&self) -> (Vec3, Vec3, Vec3);
    fn center(&self) -> Vec3;
}

impl Bin {
    pub fn new() -> Bin {
        Bin {
            aabb: AABB::new(),
            entries: 0,
            exits: 0,
        }
    }
}

pub fn partition_spatial<T: TriangleStorage>(
    triangles: &[T],
    aabbs: &[AABB],
    indices: [&mut [u32]; 3],
    first_index: usize,
    index_count: usize,
    split_dim: &mut i32,
    split_cost: &mut f32,
    node_aabb: &AABB,
    aabb_left: &mut AABB,
    aabb_right: &mut AABB,
) -> i32
{
    const SBVH_BIN_COUNT: usize = 256;

    let mut min_bin_cost = std::f32::INFINITY;
    let min_bin_index = -1;
    let min_bin_dim = -1;
    let mut min_bin_plane_distance = std::f32::NAN;

    for dim in 0..3 {
        let bounds_min = node_aabb.min[dim] - 0.001;
        let bounds_max = node_aabb.max[dim] + 0.001;
        let bounds_step = (bounds_max - bounds_min) / SBVH_BIN_COUNT as f32;
        let inv_bounds_delta = 1.0 / (bounds_max - bounds_min);

        let mut bins = vec![Bin::new(); SBVH_BIN_COUNT];
        for i in first_index..(first_index + index_count) {
            let triangle_id = indices[dim][i] as usize;
            let (mut v0, mut v1, mut v2) = triangles[triangle_id].vertices();

            if v0[dim] > v1[dim] {
                let temp = v0;
                v0 = v1;
                v1 = temp;
            }
            if v1[dim] > v2[dim] {
                let temp = v1;
                v1 = v2;
                v2 = temp;
            }
            if v0[dim] > v1[dim] {
                let temp = v0;
                v0 = v1;
                v1 = temp;
            }

            let vertex_min = v0[dim];
            let vertex_max = v2[dim];

            let bin_min = (SBVH_BIN_COUNT as f32 * ((aabbs[triangle_id].min[dim] - bounds_min) * inv_bounds_delta)) as i32;
            let bin_max = (SBVH_BIN_COUNT as f32 * ((aabbs[triangle_id].max[dim] - bounds_min) * inv_bounds_delta)) as i32;

            let bin_min = bin_min.max(0).min(SBVH_BIN_COUNT as i32 - 1) as usize;
            let bin_max = bin_max.max(0).min(SBVH_BIN_COUNT as i32 - 1) as usize;

            bins[bin_min].entries += 1;
            bins[bin_max].exits += 1;

            let vertices = [v0, v1, v2];

            for b in bin_min..=bin_max {
                let bin = &mut bins[b];

                let bin_left_plane = bounds_min + b as f32 * bounds_step;
                let bin_right_plane = bin_left_plane + bounds_step;

                assert!(bin.aabb.is_valid() || bin.aabb.is_empty());

                // Calculate relevant portion of the AABB with regard to the two planes that define the current Bin
                let mut new_box = AABB::new();

                // If all vertices lie on one side of either plane the AABB is empty
                if vertex_min >= bin_right_plane || vertex_max <= bin_left_plane {
                    // If all vertices lie between the two planes, the AABB is just the Triangle's entire AABB
                    continue;
                }

                if vertex_min >= bin_left_plane && vertex_max <= bin_right_plane {
                    new_box = aabbs[triangle_id];
                } else {
                    let mut intersections = [Vec3::zero(); 4];
                    let mut intersection_count = 0;

                    for k in 0..3 {
                        let vertex_k = vertices[k][dim];

                        for j in 1..3 {
                            let vertex_j = vertices[j][dim];
                            let delta_ij = vertex_j - vertex_k;

                            // Check if edge between Vertex i and j intersects the left plane
                            if (vertex_k < bin_left_plane && bin_left_plane <= vertex_j)
                            {
                                // Lerp to obtain exact intersection point
                                let t = (bin_left_plane - vertex_k) / delta_ij;
                                intersections[intersection_count] = (1.0 - t) * vertices[k] + t * vertices[j];
                                intersection_count += 1;
                            }

                            // Check if edge between Vertex i and j intersects the right plane
                            if (vertex_k < bin_right_plane && bin_right_plane <= vertex_j)
                            {
                                // Lerp to obtain exact intersection point
                                let t = (bin_right_plane - vertex_k) / delta_ij;
                                intersections[intersection_count] = (1.0 - t) * vertices[k] + t * vertices[j];
                                intersection_count += 1;
                            }
                        }
                    }

                    // There must be either 2 or 4 inersections with the two planes
                    assert!(intersection_count == 2 || intersection_count == 4);

                    // All intersection points should be included in the AABB
                    new_box = {
                        let mut aabb = AABB::new();
                        for i in 0..intersection_count {
                            aabb.grow(vertices[i]);
                        }
                        aabb
                    };

                    // If the middle vertex lies between the two planes it should be included in the AABB
                    if vertices[1][dim] >= bin_left_plane && vertices[1][dim] < bin_right_plane {
                        new_box.grow(vertices[1]);
                    }

                    // In case we have only two intersections with either plane it must be the case that
                    // either the leftmost or the rightmost vertex lies between the two planes
                    if intersection_count == 2 {
                        if vertex_max < bin_right_plane {
                            new_box.grow(vertices[2]);
                        } else {
                            new_box.grow(vertices[0]);
                        }
                    }

                    new_box.ensure_non_zero();
                }

                // Clip the AABB against the parent bounds
                bin.aabb.grow_bb(&new_box);
                bin.aabb = AABB::overlap(&bin.aabb, node_aabb);

                // AABB must be valid
                assert!(bin.aabb.is_valid() || bin.aabb.is_empty());

                // The AABB of the current Bin cannot exceed the planes of the current Bin
                let epsilon = 0.01;
                assert!(bin.aabb.min[dim] > bin_left_plane - epsilon);
                assert!(bin.aabb.max[dim] < bin_right_plane + epsilon);

                // The AABB of the current Bin cannot exceed the bounds of the Node's AABB
                assert!(bin.aabb.min[0] > node_aabb.min[0] - epsilon && bin.aabb.max[0] < node_aabb.max[0] + epsilon);
                assert!(bin.aabb.min[1] > node_aabb.min[1] - epsilon && bin.aabb.max[1] < node_aabb.max[1] + epsilon);
                assert!(bin.aabb.min[2] > node_aabb.min[2] - epsilon && bin.aabb.max[2] < node_aabb.max[2] + epsilon);
            }
        }
    }
    0
}