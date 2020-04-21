use glam::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::aabb::Bounds;
use crate::bvh_node::*;
use crate::mbvh_node::*;
use crate::{RayPacket4, AABB};
use rayon::prelude::*;

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BVH {
    pub nodes: Vec<BVHNode>,
    pub prim_indices: Vec<u32>,
}

impl BVH {
    pub fn empty() -> BVH {
        BVH {
            nodes: Vec::new(),
            prim_indices: Vec::new(),
        }
    }

    pub fn prim_count(&self) -> usize {
        self.prim_indices.len()
    }

    pub fn construct(aabbs: &[AABB]) -> Self {
        let mut nodes = vec![
            BVHNode {
                bounds: AABB::new()
            };
            aabbs.len() * 2
        ];
        let mut prim_indices = vec![0; aabbs.len()];
        for i in 0..aabbs.len() {
            prim_indices[i] = i as u32;
        }

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

        let mut root_bounds = AABB::new();

        root_bounds.left_first = 0;
        root_bounds.count = aabbs.len() as i32;
        for aabb in aabbs {
            root_bounds.grow_bb(aabb);
        }
        nodes[0].bounds = root_bounds.clone();

        let (sender, receiver) = std::sync::mpsc::channel();
        let thread_count = Arc::new(AtomicUsize::new(1));
        let handle = crossbeam::scope(|s| {
            BVHNode::subdivide_mt(
                0,
                root_bounds,
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
            if payload.index >= nodes.len() {
                panic!(
                    "Index was {} but only {} nodes available, bounds: {}",
                    payload.index,
                    nodes.len(),
                    payload.bounds
                );
            }
            nodes[payload.index].bounds = payload.bounds;
        }

        handle.unwrap();

        let node_count = pool_ptr.load(Ordering::SeqCst);
        nodes.resize(node_count, BVHNode::new());

        BVH {
            nodes,
            prim_indices,
        }
    }

    pub fn refit(&mut self, aabbs: &[AABB]) {
        for i in (0..self.nodes.len()).rev() {
            let mut aabb = AABB::new();
            if let Some(left_first) = self.nodes[i].get_left_first() {
                let count = self.nodes[i].get_count_unchecked();
                if count < 0 {
                    return;
                }

                aabb.left_first = left_first as i32;
                aabb.count = count as i32;
                if self.nodes[i].is_leaf() {
                    for i in 0..count {
                        let prim_id = self.prim_indices[(left_first + i as u32) as usize] as usize;
                        aabb.grow_bb(&aabbs[prim_id]);
                    }
                } else {
                    // Left node
                    aabb.grow_bb(&self.nodes[left_first as usize].bounds);
                    // Right node
                    aabb.grow_bb(&self.nodes[(left_first + 1) as usize].bounds);
                }
            }
            self.nodes[i].bounds = aabb;
        }
    }

    #[inline(always)]
    pub fn traverse<I, R>(
        &self,
        origin: &[f32; 3],
        direction: &[f32; 3],
        t_min: f32,
        t_max: f32,
        intersection_test: I,
    ) -> Option<R>
        where
            I: FnMut(usize, f32, f32) -> Option<(f32, R)>,
            R: Copy,
    {
        BVHNode::traverse(
            self.nodes.as_slice(),
            self.prim_indices.as_slice(),
            Vec3::from(*origin),
            Vec3::from(*direction),
            t_min,
            t_max,
            intersection_test,
        )
    }

    #[inline(always)]
    pub fn traverse_t<I>(
        &self,
        origin: &[f32; 3],
        direction: &[f32; 3],
        t_min: f32,
        t_max: f32,
        intersection_test: I,
    ) -> Option<f32>
        where
            I: FnMut(usize, f32, f32) -> Option<f32>,
    {
        BVHNode::traverse_t(
            self.nodes.as_slice(),
            self.prim_indices.as_slice(),
            Vec3::from(*origin),
            Vec3::from(*direction),
            t_min,
            t_max,
            intersection_test,
        )
    }

    #[inline(always)]
    pub fn occludes<I>(
        &self,
        origin: &[f32; 3],
        direction: &[f32; 3],
        t_min: f32,
        t_max: f32,
        intersection_test: I,
    ) -> bool
        where
            I: FnMut(usize, f32, f32) -> bool,
    {
        BVHNode::occludes(
            self.nodes.as_slice(),
            self.prim_indices.as_slice(),
            Vec3::from(*origin),
            Vec3::from(*direction),
            t_min,
            t_max,
            intersection_test,
        )
    }

    #[inline(always)]
    pub fn depth_test<I>(
        &self,
        origin: &[f32; 3],
        direction: &[f32; 3],
        t_min: f32,
        t_max: f32,
        intersection_test: I,
    ) -> (f32, u32)
        where
            I: Fn(usize, f32, f32) -> Option<(f32, u32)>,
    {
        BVHNode::depth_test(
            self.nodes.as_slice(),
            self.prim_indices.as_slice(),
            Vec3::from(*origin),
            Vec3::from(*direction),
            t_min,
            t_max,
            intersection_test,
        )
    }

    pub fn traverse4<I>(&self, packet: &mut RayPacket4, intersection_test: I)
        where
            I: FnMut(usize, &mut RayPacket4),
    {
        BVHNode::traverse4(
            self.nodes.as_slice(),
            self.prim_indices.as_slice(),
            packet,
            intersection_test,
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MBVH {
    pub nodes: Vec<BVHNode>,
    pub m_nodes: Vec<MBVHNode>,
    pub prim_indices: Vec<u32>,
}

impl MBVH {
    pub fn empty() -> MBVH {
        MBVH {
            nodes: Vec::new(),
            m_nodes: Vec::new(),
            prim_indices: Vec::new(),
        }
    }

    pub fn prim_count(&self) -> usize {
        self.prim_indices.len()
    }

    pub fn construct(bvh: &BVH) -> Self {
        let mut m_nodes = vec![MBVHNode::new(); bvh.nodes.len()];
        let mut pool_ptr = 1;

        if bvh.nodes.len() <= 4 {
            for i in 0..bvh.nodes.len() {
                let cur_node = &bvh.nodes[i];
                m_nodes[0].set_bounds_bb(i, &cur_node.bounds);
                m_nodes[0].children[i] = cur_node.bounds.left_first;
                m_nodes[0].counts[i] = cur_node.bounds.count;
            }

            return MBVH {
                nodes: bvh.nodes.clone(),
                m_nodes,
                prim_indices: bvh.prim_indices.clone(),
            };
        }

        MBVHNode::merge_nodes(
            0,
            0,
            bvh.nodes.as_slice(),
            m_nodes.as_mut_slice(),
            &mut pool_ptr,
        );

        MBVH {
            nodes: bvh.nodes.clone(),
            m_nodes,
            prim_indices: bvh.prim_indices.clone(),
        }
    }

    pub fn convert(bvh: BVH) -> MBVH {
        let nodes = bvh.nodes;
        let prim_indices = bvh.prim_indices;
        let mut m_nodes = vec![MBVHNode::new(); nodes.len()];
        let mut pool_ptr = 1;
        MBVHNode::merge_nodes(
            0,
            0,
            nodes.as_slice(),
            m_nodes.as_mut_slice(),
            &mut pool_ptr,
        );

        MBVH {
            nodes,
            m_nodes,
            prim_indices,
        }
    }

    #[inline(always)]
    pub fn traverse<I, R>(
        &self,
        origin: &[f32; 3],
        direction: &[f32; 3],
        t_min: f32,
        t_max: f32,
        intersection_test: I,
    ) -> Option<R>
        where
            I: FnMut(usize, f32, f32) -> Option<(f32, R)>,
            R: Copy,
    {
        MBVHNode::traverse(
            self.m_nodes.as_slice(),
            self.prim_indices.as_slice(),
            Vec3::from(*origin),
            Vec3::from(*direction),
            t_min,
            t_max,
            intersection_test,
        )
    }

    #[inline(always)]
    pub fn traverse_t<I>(
        &self,
        origin: &[f32; 3],
        direction: &[f32; 3],
        t_min: f32,
        t_max: f32,
        intersection_test: I,
    ) -> Option<f32>
        where
            I: FnMut(usize, f32, f32) -> Option<f32>,
    {
        MBVHNode::traverse_t(
            self.m_nodes.as_slice(),
            self.prim_indices.as_slice(),
            Vec3::from(*origin),
            Vec3::from(*direction),
            t_min,
            t_max,
            intersection_test,
        )
    }

    #[inline(always)]
    pub fn occludes<I>(
        &self,
        origin: &[f32; 3],
        direction: &[f32; 3],
        t_min: f32,
        t_max: f32,
        intersection_test: I,
    ) -> bool
        where
            I: FnMut(usize, f32, f32) -> bool,
    {
        MBVHNode::occludes(
            self.m_nodes.as_slice(),
            self.prim_indices.as_slice(),
            Vec3::from(*origin),
            Vec3::from(*direction),
            t_min,
            t_max,
            intersection_test,
        )
    }

    #[inline(always)]
    pub fn depth_test<I>(
        &self,
        origin: &[f32; 3],
        direction: &[f32; 3],
        t_min: f32,
        t_max: f32,
        depth_test: I,
    ) -> (f32, u32)
        where
            I: Fn(usize, f32, f32) -> Option<(f32, u32)>,
    {
        MBVHNode::depth_test(
            self.m_nodes.as_slice(),
            self.prim_indices.as_slice(),
            Vec3::from(*origin),
            Vec3::from(*direction),
            t_min,
            t_max,
            depth_test,
        )
    }

    #[inline(always)]
    pub fn traverse4<I>(&self, packet: &mut RayPacket4, intersection_test: I)
        where
            I: FnMut(usize, &mut RayPacket4),
    {
        MBVHNode::traverse4(
            self.m_nodes.as_slice(),
            self.prim_indices.as_slice(),
            packet,
            intersection_test,
        );
    }
}

impl From<BVH> for MBVH {
    fn from(bvh: BVH) -> Self {
        MBVH::convert(bvh)
    }
}

impl Bounds for BVH {
    fn bounds(&self) -> AABB {
        self.nodes[0].bounds.clone()
    }
}

impl Bounds for MBVH {
    fn bounds(&self) -> AABB {
        self.nodes[0].bounds.clone()
    }
}
