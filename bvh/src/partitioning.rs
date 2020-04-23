use crate::aabb::AABB;
use glam::Vec3;
use std::ops::Index;


#[derive(Debug, Copy, Clone)]
pub struct IndicesRef {
    x: *mut u32,
    y: *mut u32,
    z: *mut u32,
    len: usize,
}

impl<'a> IndicesRef {
    pub fn new(x: &'a mut [u32], y: &'a mut [u32], z: &'a mut [u32]) -> Self {
        assert!(x.len() == y.len() && x.len() == z.len());
        Self {
            x: x.as_mut_ptr(),
            y: y.as_mut_ptr(),
            z: z.as_mut_ptr(),
            len: x.len(),
        }
    }
    pub fn get(&self, dim: usize, index: usize) -> u32 {
        assert!(index < self.len, "Index was out of bounds, index: {}, len: {}", index, self.len);
        unsafe {
            match dim {
                0 => *self.x.add(index),
                1 => *self.y.add(index),
                2 => *self.z.add(index),
                _ => panic!("Invalid dimension: {}", dim)
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn swap(&self, dim: usize, a: usize, b: usize) {
        assert!(a < self.len, "a was out of bounds, a: {}, len: {}", a, self.len);
        assert!(b < self.len, "b was out of bounds, b: {}, len: {}", b, self.len);

        unsafe {
            match dim {
                0 => {
                    let value_a = *self.x.add(a);
                    let value_b = *self.x.add(b);
                    (*self.x.add(a)) = value_b;
                    (*self.x.add(b)) = value_a;
                }
                1 => {
                    let value_a = *self.y.add(a);
                    let value_b = *self.y.add(b);
                    (*self.y.add(a)) = value_b;
                    (*self.y.add(b)) = value_a;
                }
                2 => {
                    let value_a = *self.z.add(a);
                    let value_b = *self.z.add(b);
                    (*self.z.add(a)) = value_b;
                    (*self.z.add(b)) = value_a;
                }
                _ => panic!("Invalid dimension: {}", dim)
            }
        }
    }

    pub fn as_slice(&self, dim: usize) -> &[u32] {
        unsafe {
            match dim {
                0 => std::slice::from_raw_parts(self.x as *const u32, self.len),
                1 => std::slice::from_raw_parts(self.y as *const u32, self.len),
                2 => std::slice::from_raw_parts(self.z as *const u32, self.len),
                _ => panic!("Invalid dimension: {}", dim)
            }
        }
    }

    pub fn as_mut_slice(&self, dim: usize) -> &mut [u32] {
        unsafe {
            match dim {
                0 => std::slice::from_raw_parts_mut(self.x, self.len),
                1 => std::slice::from_raw_parts_mut(self.y, self.len),
                2 => std::slice::from_raw_parts_mut(self.z, self.len),
                _ => panic!("Invalid dimension: {}", dim)
            }
        }
    }

    pub fn with_offset(&self, offset: usize) -> Self {
        assert!(offset < self.len, "Offset was larger than length, offset: {}, length: {}", offset, self.len);
        unsafe {
            Self {
                x: self.x.add(offset),
                y: self.y.add(offset),
                z: self.z.add(offset),
                len: self.len - offset,
            }
        }
    }
}

impl std::ops::Index<(usize, usize)> for IndicesRef {
    type Output = u32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(index.1 < self.len, "Index was out of bounds, index: {}, len: {}", index.1, self.len);
        unsafe {
            match index.0 {
                0 => self.x.add(index.1).as_ref().unwrap(),
                1 => self.y.add(index.1).as_ref().unwrap(),
                2 => self.z.add(index.1).as_ref().unwrap(),
                _ => panic!("Invalid dimension: {}", index.0)
            }
        }
    }
}

impl std::ops::IndexMut<(usize, usize)> for IndicesRef {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        assert!(index.1 < self.len, "Index was out of bounds, index: {}, len: {}", index.1, self.len);
        unsafe {
            match index.0 {
                0 => self.x.add(index.1).as_mut().unwrap(),
                1 => self.y.add(index.1).as_mut().unwrap(),
                2 => self.z.add(index.1).as_mut().unwrap(),
                _ => panic!("Invalid dimension: {}", index.0)
            }
        }
    }
}

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
    centers: &[[f32; 3]],
    indices: &IndicesRef,
    first_index: usize,
    index_count: usize,
    split_dim: usize,
    split_id: usize,
    split: f32,
) {
    let mut temp = vec![0; indices.len()];

    for dim in 0..3 {
        if dim == split_dim {
            continue;
        }

        let mut left = 0;
        let mut right = split_id - first_index;

        for i in first_index..(first_index + index_count) {
            let mut goes_left = centers[indices[(dim, i)] as usize][split_dim] < split;

            if centers[indices[(dim, i)] as usize][split_dim] == split {
                // In case the current primitive has the same coordinate as the one we split on along the split dimension,
                // We don't know whether the primitive should go left or right.
                // In this case check all primitive indices on the left side of the split that
                // have the same split coordinate for equality with the current primitive index i

                let mut j = split_id - 1;

                while j >= first_index && centers[indices[(split_dim, j)] as usize][split_dim] == split {
                    if indices[(split_dim, j)] == indices[(dim, i)] {
                        goes_left = true;
                        break;
                    }

                    j -= 1;
                }
            }

            if goes_left {
                temp[left] = indices[(dim, i)];
                left += 1;
            } else {
                temp[right] = indices[(dim, i)];
                right += 1;
            }
        }

        // If these conditions are not met the copy below is invalid
        assert_eq!(left, split_id - first_index);
        assert_eq!(right, index_count);

        unsafe {
            let dest = indices.as_mut_slice(dim).as_mut_ptr().add(first_index);
            let source = temp.as_ptr();
            std::ptr::copy(source, dest, index_count);
        }
    }
}

pub fn partition_sah(
    aabbs: &[AABB],
    indices: &IndicesRef,
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
            aabb_left.grow_bb(&aabbs[indices[(0, index)] as usize]);
            sah[i] = aabb_left.area() * (i + 1) as f32;
        }

        for i in (1..index_count).rev() {
            let index = first_index + i;
            aabb_right.grow_bb(&aabbs[indices[(dim, index)] as usize]);
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
    indices: &IndicesRef,
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
            bounds_left[i].grow_bb(&aabbs[indices[(dim, first_index + i - 1)] as usize]);
            bounds_left[i] = AABB::overlap(&bounds_left[i], node_aabb);

            sah[i] = bounds_left[i].area() * i as f32;
        }

        for i in (1..index_count).rev() {
            bounds_right[i] = bounds_right[i + 1];
            bounds_right[i].grow_bb(&aabbs[indices[(dim, first_index + i)] as usize]);
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
    indices: &IndicesRef,
    first_index: usize,
    index_count: usize,
    split_dim: &mut i32,
    split_cost: &mut f32,
    plane_distance: &mut f32,
    node_aabb: &AABB,
    aabb_left: &mut AABB,
    aabb_right: &mut AABB,
    n_left: &mut i32,
    n_right: &mut i32,
) -> i32
{
    const SBVH_BIN_COUNT: usize = 256;

    let mut min_bin_cost = std::f32::INFINITY;
    let mut min_bin_index = -1;
    let mut min_bin_dim = -1;
    let mut min_bin_plane_distance = std::f32::NAN;

    for dim in 0..3 {
        let bounds_min = node_aabb.min[dim] - 0.001;
        let bounds_max = node_aabb.max[dim] + 0.001;
        let bounds_step = (bounds_max - bounds_min) / SBVH_BIN_COUNT as f32;
        let inv_bounds_delta = 1.0 / (bounds_max - bounds_min);

        let mut bins = vec![Bin::new(); SBVH_BIN_COUNT];
        for i in first_index..(first_index + index_count) {
            let triangle_id = indices[(dim, i)] as usize;
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
                            if vertex_k < bin_left_plane && bin_left_plane <= vertex_j
                            {
                                // Lerp to obtain exact intersection point
                                let t = (bin_left_plane - vertex_k) / delta_ij;
                                intersections[intersection_count] = (1.0 - t) * vertices[k] + t * vertices[j];
                                intersection_count += 1;
                            }

                            // Check if edge between Vertex i and j intersects the right plane
                            if vertex_k < bin_right_plane && bin_right_plane <= vertex_j
                            {
                                // Lerp to obtain exact intersection point
                                let t = (bin_right_plane - vertex_k) / delta_ij;
                                intersections[intersection_count] = (1.0 - t) * vertices[k] + t * vertices[j];
                                intersection_count += 1;
                            }
                        }
                    }

                    // There must be either 2 or 4 intersections with the two planes
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

        let mut bin_sah = vec![0.0; SBVH_BIN_COUNT];
        let mut bounds_left = vec![AABB::new(); SBVH_BIN_COUNT];
        let mut bounds_right = vec![AABB::new(); SBVH_BIN_COUNT + 1];

        let mut count_left = vec![0; SBVH_BIN_COUNT];
        let mut count_right = vec![0; SBVH_BIN_COUNT + 1];

        for b in 1..SBVH_BIN_COUNT {
            bounds_left[b] = bounds_left[b - 1];
            bounds_left[b].grow_bb(&bins[b - 1].aabb);
            assert!(bounds_left[b].is_valid() || bounds_left[b].is_empty());

            count_left[b] = count_left[b - 1] + bins[b - 1].entries;

            if count_left[b] < index_count as u32 {
                bin_sah[b] = bounds_left[b].area() * count_left[b] as f32;
            } else {
                bin_sah[b] = std::f32::INFINITY;
            }
        }

        for b in 1..SBVH_BIN_COUNT {
            bounds_right[b] = bounds_right[b + 1];
            bounds_right[b].grow_bb(&bins[b].aabb);
            assert!(bounds_right[b].is_valid() || bounds_right[b].is_empty());

            count_right[b] = count_right[b + 1] + bins[b].exits;

            if count_right[b] < index_count as u32 {
                bin_sah[b] += bounds_right[b].area() * count_right[b] as f32;
            } else {
                bin_sah[b] = std::f32::INFINITY;
            }
        }

        assert_eq!(count_left[SBVH_BIN_COUNT - 1] + bins[SBVH_BIN_COUNT - 1].entries, index_count as u32);
        assert_eq!(count_right[1] + bins[0].exits, index_count as u32);

        for b in 1..SBVH_BIN_COUNT {
            let cost = bin_sah[b];

            if cost < min_bin_cost {
                min_bin_cost = cost;
                min_bin_index = b as i32;
                min_bin_dim = dim as i32;

                *aabb_left = bounds_left[b];
                *aabb_right = bounds_right[b];

                *n_left = count_left[b] as i32;
                *n_right = count_right[b] as i32;

                min_bin_plane_distance = bounds_min + bounds_step * b as f32;
            }
        }
    }

    *split_dim = min_bin_dim;
    *split_cost = min_bin_cost;
    *plane_distance = min_bin_plane_distance;

    min_bin_index
}