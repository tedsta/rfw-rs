use crate::bvh_node::BVHNode;
use crate::{BVHBuilder, Aabb, BVHResult};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use crossbeam::thread::Scope;
use std::sync::atomic::*;

pub struct TopDownBuilderBinnedSAH {
    bins: u32,
    max_depth: u32,
    max_prims: u32,
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
    pub fn subdivide(
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
}

impl BVHBuilder for TopDownBuilderBinnedSAH {
    fn build(&self, aabbs: &[Aabb]) -> BVHResult {
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

        BVHResult {
            nodes,
            prim_indices,
        }
    }
}
