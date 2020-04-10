pub mod top_down_builder;
pub mod hlbvh_builder;

pub use top_down_builder::*;
pub use hlbvh_builder::*;

use crate::{Aabb, BVHNode};
use std::sync::Arc;

pub struct BVHResult {
    pub nodes: Vec<BVHNode>,
    pub prim_indices: Vec<u32>,
}

pub trait BVHBuilder {
    fn build(&self, aabbs: &[Aabb]) -> BVHResult;
}
