use crate::{BVHBuilder, BVHResult, Aabb, BVHNode};
use rayon::prelude::*;

pub struct BottomUpBuilderSAH {
    bins: u32,
    max_depth: u32,
    max_prims: u32,
}

struct PrimitiveInfo {
    pub prim_number: u32,
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub centroid: [f32; 3],
}

struct BVHBuildNode {
    pub min: [f32; 3],
    pub max: [f32; 3],
    children: [Box<BVHBuildNode>; 2],
    pub split_axis: i32,
    pub first_prim_offset: i32,
    pub n_primitives: i32,
}

struct MortonPrimitive {
    pub prim_index: i32,
    pub morton_code: u32,
}

struct LBVHTreelet {
    pub start_index: i32,
    pub n_primitives: i32,
    pub build_node: Box<BVHBuildNode>,
}

struct LinearBVHNode {
    pub min: [f32; 3],
    pub max: [f32; 3],

}

impl BVHBuilder for BottomUpBuilderSAH {
    fn build(&self, aabbs: &[Aabb]) -> BVHResult {
        let mut nodes = vec![BVHNode::new(); aabbs.len() * 2];

        let prim_info: Vec<PrimitiveInfo> = aabbs.iter().enumerate().map(|(i, aabb)| {
            PrimitiveInfo {
                prim_number: i as u32,
                min: aabb.min,
                max: aabb.max,
                centroid: aabb.center().into(),
            }
        }).collect();

        let prim_indices = Vec::new();

        BVHResult {
            nodes,
            prim_indices,
        }
    }
}