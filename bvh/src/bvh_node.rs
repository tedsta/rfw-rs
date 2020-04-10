use glam::*;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use serde::{Serialize, Deserialize};

use crate::{RayPacket4, Aabb};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct BVHNode {
    pub bounds: Aabb,
    pub left_first: i32,
    pub count: i32,
}


impl Display for BVHNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.bounds)
    }
}

#[allow(dead_code)]
impl BVHNode {
    pub fn new() -> BVHNode {
        BVHNode {
            bounds: Aabb::new(),
            left_first: -1,
            count: -1,
        }
    }

    pub fn depth_test<I>(
        tree: &[BVHNode],
        prim_indices: &[u32],
        origin: Vec3,
        dir: Vec3,
        t_min: f32,
        t_max: f32,
        depth_test: I,
    ) -> (f32, u32)
        where
            I: Fn(usize, f32, f32) -> Option<(f32, u32)>,
    {
        let mut t = t_max;
        let dir_inverse = Vec3::new(1.0, 1.0, 1.0) / dir;

        if tree[0].bounds.intersect(origin, dir_inverse, t).is_none() {
            return (t_max, 0);
        }

        let mut depth: i32 = 0;
        let mut hit_stack = [0; 64];
        let mut stack_ptr: i32 = 0;

        while stack_ptr >= 0 {
            depth = depth + 1;
            let node = &tree[hit_stack[stack_ptr as usize] as usize];
            stack_ptr = stack_ptr - 1;

            if node.count > -1 {
                // Leaf node
                for i in 0..node.count {
                    let prim_id = prim_indices[(node.left_first + i) as usize];
                    if let Some((new_t, d)) = depth_test(prim_id as usize, t_min, t) {
                        t = new_t;
                        depth += d as i32;
                    }
                }
            } else if node.left_first >= 0 {
                let hit_left =
                    tree[node.left_first as usize]
                        .bounds
                        .intersect(origin, dir_inverse, t);
                let hit_right = tree[(node.left_first + 1) as usize]
                    .bounds
                    .intersect(origin, dir_inverse, t);
                let new_stack_ptr = Self::sort_nodes(
                    hit_left,
                    hit_right,
                    hit_stack.as_mut(),
                    stack_ptr,
                    node.left_first,
                );
                stack_ptr = new_stack_ptr;
            }
        }

        (t, depth as u32)
    }

    pub fn traverse<I, R>(
        tree: &[BVHNode],
        prim_indices: &[u32],
        origin: Vec3,
        dir: Vec3,
        t_min: f32,
        t_max: f32,
        mut intersection_test: I,
    ) -> Option<R>
        where
            I: FnMut(usize, f32, f32) -> Option<(f32, R)>,
            R: Copy,
    {
        let mut hit_stack = [0; 64];
        let mut stack_ptr: i32 = 0;
        let mut t = t_max;
        let mut hit_record = None;

        let dir_inverse = Vec3::new(1.0, 1.0, 1.0) / dir;
        hit_stack[stack_ptr as usize] = 0;
        while stack_ptr >= 0 {
            let node = &tree[hit_stack[stack_ptr as usize] as usize];
            stack_ptr = stack_ptr - 1;

            if node.count > -1 {
                // Leaf node
                for i in 0..node.count {
                    let prim_id = prim_indices[(node.left_first + i) as usize];
                    if let Some((new_t, new_hit)) = intersection_test(prim_id as usize, t_min, t) {
                        t = new_t;
                        hit_record = Some(new_hit);
                    }
                }
            } else if node.left_first >= 0 {
                let hit_left =
                    tree[node.left_first as usize]
                        .bounds
                        .intersect(origin, dir_inverse, t);
                let hit_right = tree[(node.left_first + 1) as usize]
                    .bounds
                    .intersect(origin, dir_inverse, t);
                stack_ptr = Self::sort_nodes(
                    hit_left,
                    hit_right,
                    hit_stack.as_mut(),
                    stack_ptr,
                    node.left_first,
                );
            }
        }

        hit_record
    }

    pub fn traverse_t<I>(
        tree: &[BVHNode],
        prim_indices: &[u32],
        origin: Vec3,
        dir: Vec3,
        t_min: f32,
        t_max: f32,
        mut intersection_test: I,
    ) -> Option<f32>
        where
            I: FnMut(usize, f32, f32) -> Option<f32>,
    {
        let mut hit_stack = [0; 64];
        let mut stack_ptr: i32 = 0;
        let mut t = t_max;

        let dir_inverse = Vec3::new(1.0, 1.0, 1.0) / dir;
        hit_stack[stack_ptr as usize] = 0;
        while stack_ptr >= 0 {
            let node = &tree[hit_stack[stack_ptr as usize] as usize];
            stack_ptr = stack_ptr - 1;

            if node.count > -1 {
                // Leaf node
                for i in 0..node.count {
                    let prim_id = prim_indices[(node.left_first + i) as usize];
                    if let Some(new_t) = intersection_test(prim_id as usize, t_min, t) {
                        t = new_t;
                    }
                }
            } else if node.left_first >= 0 {
                let hit_left =
                    tree[node.left_first as usize]
                        .bounds
                        .intersect(origin, dir_inverse, t);
                let hit_right = tree[(node.left_first + 1) as usize]
                    .bounds
                    .intersect(origin, dir_inverse, t);
                stack_ptr = Self::sort_nodes(
                    hit_left,
                    hit_right,
                    hit_stack.as_mut(),
                    stack_ptr,
                    node.left_first,
                );
            }
        }

        if t < t_max {
            Some(t)
        } else {
            None
        }
    }

    pub fn occludes<I>(
        tree: &[BVHNode],
        prim_indices: &[u32],
        origin: Vec3,
        dir: Vec3,
        t_min: f32,
        t_max: f32,
        mut intersection_test: I,
    ) -> bool
        where
            I: FnMut(usize, f32, f32) -> bool,
    {
        let mut hit_stack = [0; 64];
        let mut stack_ptr: i32 = 0;

        let dir_inverse = Vec3::new(1.0, 1.0, 1.0) / dir;
        hit_stack[stack_ptr as usize] = 0;
        while stack_ptr >= 0 {
            let node = &tree[hit_stack[stack_ptr as usize] as usize];
            stack_ptr = stack_ptr - 1;

            if node.count > -1 {
                // Leaf node
                for i in 0..node.count {
                    let prim_id = prim_indices[(node.left_first + i) as usize];
                    if intersection_test(prim_id as usize, t_min, t_max) {
                        return true;
                    }
                }
            } else if node.left_first >= 0 {
                let hit_left = tree[node.left_first as usize].bounds.intersect(
                    origin,
                    dir_inverse,
                    t_max,
                );
                let hit_right = tree[(node.left_first + 1) as usize]
                    .bounds
                    .intersect(origin, dir_inverse, t_max);
                stack_ptr = Self::sort_nodes(
                    hit_left,
                    hit_right,
                    hit_stack.as_mut(),
                    stack_ptr,
                    node.left_first,
                );
            }
        }

        false
    }

    fn sort_nodes(
        left: Option<(f32, f32)>,
        right: Option<(f32, f32)>,
        hit_stack: &mut [i32],
        mut stack_ptr: i32,
        left_first: i32,
    ) -> i32 {
        if left.is_some() & &right.is_some() {
            let (t_near_left, _) = left.unwrap();
            let (t_near_right, _) = right.unwrap();

            if t_near_left < t_near_right {
                stack_ptr = stack_ptr + 1;
                hit_stack[stack_ptr as usize] = left_first;
                stack_ptr = stack_ptr + 1;
                hit_stack[stack_ptr as usize] = left_first + 1;
            } else {
                stack_ptr = stack_ptr + 1;
                hit_stack[stack_ptr as usize] = left_first + 1;
                stack_ptr = stack_ptr + 1;
                hit_stack[stack_ptr as usize] = left_first;
            }
        } else if left.is_some() {
            stack_ptr = stack_ptr + 1;
            hit_stack[stack_ptr as usize] = left_first;
        } else if right.is_some() {
            stack_ptr = stack_ptr + 1;
            hit_stack[stack_ptr as usize] = left_first + 1;
        }

        stack_ptr
    }

    fn sort_nodes4(
        left: Option<[f32; 4]>,
        right: Option<[f32; 4]>,
        hit_stack: &mut [i32],
        mut stack_ptr: i32,
        left_first: i32,
    ) -> i32 {
        if left.is_some() & &right.is_some() {
            let t_near_left = Vec4::from(left.unwrap());
            let t_near_right = Vec4::from(right.unwrap());

            if t_near_left.cmplt(t_near_right).bitmask() > 0 {
                stack_ptr = stack_ptr + 1;
                hit_stack[stack_ptr as usize] = left_first;
                stack_ptr = stack_ptr + 1;
                hit_stack[stack_ptr as usize] = left_first + 1;
            } else {
                stack_ptr = stack_ptr + 1;
                hit_stack[stack_ptr as usize] = left_first + 1;
                stack_ptr = stack_ptr + 1;
                hit_stack[stack_ptr as usize] = left_first;
            }
        } else if left.is_some() {
            stack_ptr = stack_ptr + 1;
            hit_stack[stack_ptr as usize] = left_first;
        } else if right.is_some() {
            stack_ptr = stack_ptr + 1;
            hit_stack[stack_ptr as usize] = left_first + 1;
        }

        stack_ptr
    }

    pub fn traverse4<I: FnMut(usize, &mut RayPacket4)>(
        tree: &[BVHNode],
        prim_indices: &[u32],
        packet: &mut RayPacket4,
        mut intersection_test: I,
    )
    {
        let mut hit_stack = [0; 64];
        let mut stack_ptr: i32 = 0;

        let one = Vec4::one();
        let inv_dir_x = one / Vec4::from(packet.direction_x);
        let inv_dir_y = one / Vec4::from(packet.direction_y);
        let inv_dir_z = one / Vec4::from(packet.direction_z);

        while stack_ptr >= 0 {
            let node = &tree[hit_stack[stack_ptr as usize] as usize];
            stack_ptr = stack_ptr - 1;

            if node.count > -1 {
                // Leaf node
                for i in 0..node.count {
                    let prim_id = prim_indices[(node.left_first + i) as usize] as usize;
                    intersection_test(prim_id, packet);
                }
            } else if node.left_first >= 0 {
                let hit_left =
                    tree[node.left_first as usize].bounds.intersect4(packet, inv_dir_x, inv_dir_y, inv_dir_z);
                let hit_right = tree[(node.left_first + 1) as usize].bounds.intersect4(packet, inv_dir_x, inv_dir_y, inv_dir_z);

                stack_ptr = Self::sort_nodes4(
                    hit_left,
                    hit_right,
                    hit_stack.as_mut(),
                    stack_ptr,
                    node.left_first,
                );
            }
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.count >= 0
    }
}
