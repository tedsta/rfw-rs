use crate::{BVHBuilder, Aabb, RayPacket4, Ray, Bounds};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use crossbeam::thread::Scope;
use std::sync::atomic::*;
use std::fmt::{Display, Formatter};
use glam::*;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
struct BVHNode {
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

    pub fn is_leaf(&self) -> bool {
        self.count >= 0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopDownBuilderBinnedSAH {
    nodes: Vec<BVHNode>,
    prim_indices: Vec<u32>,
    bins: u32,
    max_depth: u32,
    max_prims: u32,
}

impl Bounds for TopDownBuilderBinnedSAH {
    fn bounds(&self) -> Aabb {
        self.nodes[0].bounds.clone()
    }
}

struct NewNodeInfo {
    pub left_box: Aabb,
    pub left_left_first: i32,
    pub left_count: i32,
    pub right_box: Aabb,
    pub right_left_first: i32,
    pub right_count: i32,
    pub left: u32,
}

struct NodeUpdatePayLoad {
    pub bounds: Aabb,
    pub left_first: i32,
    pub count: i32,
    pub index: u32,
}

impl TopDownBuilderBinnedSAH {
    pub fn new(bins: usize, max_depth: usize, max_prims: usize) -> Self {
        Self {
            nodes: Vec::new(),
            prim_indices: Vec::new(),
            bins: bins as u32,
            max_depth: max_depth as u32,
            max_prims: max_prims as u32,
        }
    }

    fn subdivide_mt<'a>(
        builder: &'a Self,
        index: usize,
        mut bounds: Aabb,
        mut left_first: i32,
        mut count: i32,
        aabbs: &'a [Aabb],
        centers: &'a [[f32; 3]],
        update_node: Sender<NodeUpdatePayLoad>,
        prim_indices: &'a mut [u32],
        depth: u32,
        pool_ptr: Arc<AtomicUsize>,
        thread_count: Arc<AtomicUsize>,
        max_threads: usize,
        scope: &Scope<'a>,
    ) {
        let depth = depth + 1;
        if depth >= builder.max_depth {
            update_node
                .send(NodeUpdatePayLoad { index: index as u32, bounds, left_first, count })
                .unwrap();
            return;
        }

        let new_nodes = Self::partition_atomic(
            builder,
            &bounds,
            left_first,
            count,
            aabbs,
            centers,
            prim_indices,
            pool_ptr.clone(),
        );
        if new_nodes.is_none() {
            return;
        }

        let new_nodes = new_nodes.unwrap();
        left_first = new_nodes.left as i32;
        count = -1;

        let (left_indices, right_indices) =
            prim_indices.split_at_mut(new_nodes.left_count as usize);
        let threads = thread_count.load(Ordering::SeqCst);

        let mut handle = None;
        update_node
            .send(NodeUpdatePayLoad { index: index as u32, bounds, left_first, count })
            .unwrap();

        if new_nodes.left_count > builder.max_prims as i32 {
            let left = new_nodes.left as usize;
            let left_box = new_nodes.left_box;
            let sender = update_node.clone();
            let tc = thread_count.clone();
            let pp = pool_ptr.clone();

            let left_first = new_nodes.left_left_first;
            let left_count = new_nodes.left_count;

            if threads < num_cpus::get() {
                thread_count.fetch_add(1, Ordering::SeqCst);
                handle = Some(scope.spawn(move |s| {
                    Self::subdivide_mt(
                        builder,
                        left,
                        left_box,
                        left_first,
                        left_count,
                        aabbs,
                        centers,
                        sender,
                        left_indices,
                        depth,
                        pp,
                        tc,
                        max_threads,
                        s,
                    );
                }));
            } else {
                Self::subdivide_mt(
                    builder,
                    left,
                    left_box,
                    new_nodes.left_left_first,
                    new_nodes.left_count,
                    aabbs,
                    centers,
                    sender,
                    left_indices,
                    depth,
                    pp,
                    tc,
                    max_threads,
                    scope,
                );
            }
        } else {
            update_node
                .send(NodeUpdatePayLoad {
                    index: new_nodes.left as u32,
                    bounds: new_nodes.left_box,
                    left_first: new_nodes.left_left_first,
                    count: new_nodes.left_count,
                })
                .unwrap();
        }

        if new_nodes.right_count > builder.max_prims as i32 {
            let right = (new_nodes.left + 1) as usize;
            let right_box = new_nodes.right_box;

            Self::subdivide_mt(
                builder,
                right,
                right_box,
                new_nodes.right_left_first,
                new_nodes.right_count,
                aabbs,
                centers,
                update_node,
                right_indices,
                depth,
                pool_ptr,
                thread_count.clone(),
                max_threads,
                scope,
            );
        } else {
            update_node
                .send(NodeUpdatePayLoad {
                    index: new_nodes.left + 1,
                    bounds: new_nodes.right_box,
                    left_first: new_nodes.right_left_first,
                    count: new_nodes.right_count,
                })
                .unwrap();
        }

        if let Some(handle) = handle {
            handle.join().unwrap();
            thread_count.fetch_sub(1, Ordering::SeqCst);
        }
    }

    // Reference single threaded subdivide method
    fn subdivide(
        builder: &Self,
        index: usize,
        aabbs: &[Aabb],
        left_first: i32,
        count: i32,
        centers: &[[f32; 3]],
        tree: &mut [BVHNode],
        prim_indices: &mut [u32],
        depth: u32,
        pool_ptr: &mut usize,
    ) {
        let depth = depth + 1;
        if depth >= builder.max_depth {
            return;
        }

        let new_nodes = Self::partition(
            builder,
            &tree[index].bounds,
            left_first,
            count,
            aabbs,
            centers,
            prim_indices,
            pool_ptr,
        );
        if new_nodes.is_none() {
            return;
        }
        let new_nodes = new_nodes.unwrap();

        tree[index].left_first = new_nodes.left as i32;
        tree[index].count = -1;

        let (left_indices, right_indices) =
            prim_indices.split_at_mut(new_nodes.left_count as usize);

        let left = new_nodes.left as usize;

        tree[left].bounds = new_nodes.left_box;
        if tree[left].count > builder.max_prims as i32 {
            Self::subdivide(
                builder,
                left,
                aabbs,
                new_nodes.left_left_first,
                new_nodes.left_count,
                centers,
                tree,
                left_indices,
                depth,
                pool_ptr,
            );
        }

        let right = (new_nodes.left + 1) as usize;

        tree[right].bounds = new_nodes.right_box;
        if tree[right].count > builder.max_prims as i32 {
            Self::subdivide(
                builder,
                right,
                aabbs,
                new_nodes.right_left_first,
                new_nodes.right_count,
                centers,
                tree,
                right_indices,
                depth,
                pool_ptr,
            );
        }
    }


    fn partition_atomic(
        builder: &Self,
        bounds: &Aabb,
        left_first: i32,
        count: i32,
        aabbs: &[Aabb],
        centers: &[[f32; 3]],
        prim_indices: &mut [u32],
        pool_ptr: Arc<AtomicUsize>,
    ) -> Option<NewNodeInfo> {
        let mut best_split = 0.0 as f32;
        let mut best_axis = 0;

        let mut best_left_box = Aabb::new();
        let mut best_right_box = Aabb::new();

        let mut lowest_cost = 1e34;
        let parent_cost = bounds.area() * count as f32;
        let lengths = bounds.lengths();

        let bin_size = 1.0 / (builder.bins + 2) as f32;

        for i in 1..(builder.bins + 2) {
            let bin_offset = i as f32 * bin_size;
            for axis in 0..3 {
                let split_offset = bounds.min[axis] + lengths[axis] * bin_offset;

                let mut left_count = 0;
                let mut right_count = 0;

                let mut left_box = Aabb::new();
                let mut right_box = Aabb::new();

                let (left_area, right_area) = {
                    for idx in 0..count {
                        let idx = unsafe { *prim_indices.get_unchecked(idx as usize) as usize };
                        let center = centers[idx][axis];
                        let aabb = unsafe { aabbs.get_unchecked(idx) };

                        if center <= split_offset {
                            left_box.grow_bb(aabb);
                            left_count = left_count + 1;
                        } else {
                            right_box.grow_bb(aabb);
                            right_count = right_count + 1;
                        }
                    }

                    (left_box.area(), right_box.area())
                };

                let split_node_cost =
                    left_area * left_count as f32 + right_area * right_count as f32;
                if lowest_cost > split_node_cost {
                    lowest_cost = split_node_cost;
                    best_split = split_offset;
                    best_axis = axis;
                    best_left_box = left_box;
                    best_right_box = right_box;
                }
            }
        }

        if parent_cost < lowest_cost {
            return None;
        }

        let mut left_count = 0;

        for idx in 0..count {
            let id = unsafe { *prim_indices.get_unchecked(idx as usize) as usize };
            let center = centers[id][best_axis];

            if center <= best_split {
                prim_indices.swap((idx) as usize, (left_count) as usize);
                left_count = left_count + 1;
            }
        }

        let right_first = left_first + left_count;
        let right_count = count - left_count;

        let left = pool_ptr.fetch_add(2, Ordering::SeqCst);

        let left_left_first = left_first;
        let left_count = left_count;
        best_left_box.offset_by(1e-6);

        let right_left_first = right_first;
        best_right_box.offset_by(1e-6);

        Some(NewNodeInfo {
            left: left as u32,
            left_box: best_left_box,
            left_left_first,
            left_count,
            right_box: best_right_box,
            right_left_first,
            right_count,
        })
    }

    fn partition(
        builder: &Self,
        bounds: &Aabb,
        left_first: i32,
        count: i32,
        aabbs: &[Aabb],
        centers: &[[f32; 3]],
        prim_indices: &mut [u32],
        pool_ptr: &mut usize,
    ) -> Option<NewNodeInfo> {
        let mut best_split = 0.0 as f32;
        let mut best_axis = 0;

        let mut best_left_box = Aabb::new();
        let mut best_right_box = Aabb::new();

        let mut lowest_cost = 1e34;
        let parent_cost = bounds.area() * count as f32;
        let lengths = bounds.lengths();

        let bin_size = 1.0 / (builder.bins + 2) as f32;

        for i in 1..(builder.bins + 2) {
            let bin_offset = i as f32 * bin_size;
            for axis in 0..3 {
                let split_offset = bounds.min[axis] + lengths[axis] * bin_offset;

                let mut left_count = 0;
                let mut right_count = 0;

                let mut left_box = Aabb::new();
                let mut right_box = Aabb::new();

                let (left_area, right_area) = {
                    for idx in 0..count {
                        let idx = unsafe { *prim_indices.get_unchecked(idx as usize) as usize };
                        let center = centers[idx][axis];
                        let aabb = unsafe { aabbs.get_unchecked(idx) };

                        if center <= split_offset {
                            left_box.grow_bb(aabb);
                            left_count = left_count + 1;
                        } else {
                            right_box.grow_bb(aabb);
                            right_count = right_count + 1;
                        }
                    }

                    (left_box.area(), right_box.area())
                };

                let split_node_cost =
                    left_area * left_count as f32 + right_area * right_count as f32;
                if lowest_cost > split_node_cost {
                    lowest_cost = split_node_cost;
                    best_split = split_offset;
                    best_axis = axis;
                    best_left_box = left_box;
                    best_right_box = right_box;
                }
            }
        }

        if parent_cost < lowest_cost {
            return None;
        }

        let mut left_count = 0;

        for idx in 0..count {
            let id = unsafe { *prim_indices.get_unchecked(idx as usize) as usize };
            let center = centers[id][best_axis];

            if center <= best_split {
                prim_indices.swap((idx) as usize, (left_count) as usize);
                left_count = left_count + 1;
            }
        }

        let right_first = left_first + left_count;
        let right_count = count - left_count;

        let left = *pool_ptr;
        *pool_ptr += 2;

        let left_left_first = left_first;
        let left_count = left_count;
        best_left_box.offset_by(1e-6);

        let right_left_first = right_first;
        let right_count = right_count;
        best_right_box.offset_by(1e-6);

        Some(NewNodeInfo {
            left: left as u32,
            left_box: best_left_box,
            left_left_first,
            left_count,
            right_box: best_right_box,
            right_left_first,
            right_count,
        })
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
}

impl BVHBuilder for TopDownBuilderBinnedSAH {
    fn build(&mut self, aabbs: &[Aabb]) {
        let mut nodes = vec![BVHNode::new(); aabbs.len() * 2];
        let mut prim_indices = vec![0; aabbs.len()];
        prim_indices.iter_mut().enumerate().for_each(|(i, prim)| { *prim = i as u32 });

        let centers = aabbs
            .into_par_iter()
            .map(|bb| {
                let center = [
                    (bb.min[0] + bb.max[0]) * 0.5,
                    (bb.min[1] + bb.max[1]) * 0.5,
                    (bb.min[2] + bb.max[2]) * 0.5,
                ];
                center
            })
            .collect::<Vec<[f32; 3]>>();
        let pool_ptr = Arc::new(AtomicUsize::new(2));
        let depth = 1;

        let left_first = 0;
        let count = aabbs.len() as i32;
        for aabb in aabbs {
            nodes[0].bounds.grow_bb(aabb);
        }

        nodes[0].left_first = left_first;
        nodes[0].count = count;

        let (sender, receiver) = std::sync::mpsc::channel();
        let thread_count = Arc::new(AtomicUsize::new(1));
        let handle = crossbeam::scope(|s| {
            Self::subdivide_mt(
                self,
                0,
                nodes[0].bounds.clone(),
                left_first,
                count,
                aabbs,
                &centers,
                sender,
                prim_indices.as_mut_slice(),
                depth,
                pool_ptr.clone(),
                thread_count,
                num_cpus::get(),
                s,
            );
        });

        for payload in receiver.iter() {
            if payload.index as usize >= nodes.len() {
                panic!(
                    "Index was {} but only {} nodes available",
                    payload.index,
                    nodes.len()
                );
            }

            nodes[payload.index as usize].bounds = payload.bounds;
            nodes[payload.index as usize].left_first = payload.left_first;
            nodes[payload.index as usize].count = payload.count;
        }

        handle.unwrap();

        let node_count = pool_ptr.load(Ordering::SeqCst);
        nodes.resize(node_count, BVHNode::new());

        self.nodes = nodes;
        self.prim_indices = prim_indices;
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn traverse_t<I: FnMut(usize, f32, f32) -> Option<f32>>(&self, ray: Ray, t_min: f32, t_max: f32, mut is: I) -> Option<f32> {
        let mut hit_stack = [0; 64];
        let mut stack_ptr: i32 = 0;
        let mut t = t_max;

        let (origin, direction) = ray.into();
        let dir_inverse = Vec3::new(1.0, 1.0, 1.0) / direction;
        hit_stack[stack_ptr as usize] = 0;
        while stack_ptr >= 0 {
            let node = &self.nodes[hit_stack[stack_ptr as usize] as usize];
            stack_ptr = stack_ptr - 1;

            if node.count > -1 {
                // Leaf node
                for i in 0..node.count {
                    let prim_id = self.prim_indices[(node.left_first + i) as usize];
                    if let Some(new_t) = is(prim_id as usize, t_min, t) {
                        t = new_t;
                    }
                }
            } else if node.left_first >= 0 {
                let hit_left =
                    self.nodes[node.left_first as usize]
                        .bounds
                        .intersect(origin, dir_inverse, t);
                let hit_right = self.nodes[(node.left_first + 1) as usize]
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

    fn occludes<I: FnMut(usize, f32, f32) -> bool>(&self, ray: Ray, t_min: f32, t_max: f32, mut occludes: I) -> bool {
        let mut hit_stack = [0; 64];
        let mut stack_ptr: i32 = 0;

        let (origin, direction) = ray.into();
        let dir_inverse = Vec3::new(1.0, 1.0, 1.0) / direction;
        hit_stack[stack_ptr as usize] = 0;
        while stack_ptr >= 0 {
            let node = &self.nodes[hit_stack[stack_ptr as usize] as usize];
            stack_ptr = stack_ptr - 1;

            if node.count > -1 {
                // Leaf node
                for i in 0..node.count {
                    let prim_id = self.prim_indices[(node.left_first + i) as usize];
                    if occludes(prim_id as usize, t_min, t_max) {
                        return true;
                    }
                }
            } else if node.left_first >= 0 {
                let hit_left = self.nodes[node.left_first as usize].bounds.intersect(
                    origin,
                    dir_inverse,
                    t_max,
                );
                let hit_right = self.nodes[(node.left_first + 1) as usize]
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

    fn depth_test<I: Fn(usize, f32, f32) -> Option<(f32, u32)>>(&self, ray: Ray, t_min: f32, t_max: f32, is: I) -> (f32, u32) {
        let mut t = t_max;

        let (origin, direction) = ray.into();
        let dir_inverse = Vec3::new(1.0, 1.0, 1.0) / direction;

        if self.nodes[0].bounds.intersect(origin, dir_inverse, t).is_none() {
            return (t_max, 0);
        }

        let mut depth: i32 = 0;
        let mut hit_stack = [0; 64];
        let mut stack_ptr: i32 = 0;

        while stack_ptr >= 0 {
            depth = depth + 1;
            let node = &self.nodes[hit_stack[stack_ptr as usize] as usize];
            stack_ptr = stack_ptr - 1;

            if node.count > -1 {
                // Leaf node
                for i in 0..node.count {
                    let prim_id = self.prim_indices[(node.left_first + i) as usize];
                    if let Some((new_t, d)) = is(prim_id as usize, t_min, t) {
                        t = new_t;
                        depth += d as i32;
                    }
                }
            } else if node.left_first >= 0 {
                let hit_left =
                    self.nodes[node.left_first as usize]
                        .bounds
                        .intersect(origin, dir_inverse, t);
                let hit_right = self.nodes[(node.left_first + 1) as usize]
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

    fn traverse4<I: FnMut(usize, &mut RayPacket4)>(&self, packet: &mut RayPacket4, mut is: I) {
        let mut hit_stack = [0; 64];
        let mut stack_ptr: i32 = 0;

        let one = Vec4::one();
        let inv_dir_x = one / Vec4::from(packet.direction_x);
        let inv_dir_y = one / Vec4::from(packet.direction_y);
        let inv_dir_z = one / Vec4::from(packet.direction_z);

        while stack_ptr >= 0 {
            let node = &self.nodes[hit_stack[stack_ptr as usize] as usize];
            stack_ptr = stack_ptr - 1;

            if node.count > -1 {
                // Leaf node
                for i in 0..node.count {
                    let prim_id = self.prim_indices[(node.left_first + i) as usize] as usize;
                    is(prim_id, packet);
                }
            } else if node.left_first >= 0 {
                let hit_left =
                    self.nodes[node.left_first as usize].bounds.intersect4(packet, inv_dir_x, inv_dir_y, inv_dir_z);
                let hit_right = self.nodes[(node.left_first + 1) as usize].bounds.intersect4(packet, inv_dir_x, inv_dir_y, inv_dir_z);

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
}
