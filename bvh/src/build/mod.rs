pub mod top_down_builder;
pub mod lbvh_builder;
pub mod lbvh;

pub use top_down_builder::*;
pub use lbvh_builder::*;
pub use lbvh::*;

use crate::{Aabb, RayPacket4, Ray, Bounds};
use std::sync::Arc;

pub trait BVHBuilder : Bounds {
    fn build(&mut self, aabbs: &[Aabb]);
    fn len(&self) -> usize;
    // fn refit(&mut self, aabbs: &[Aabb]);

    fn traverse_t<I: FnMut(usize, f32, f32) -> Option<f32>>(&self, ray: Ray, t_min: f32, t_max: f32, i: I) -> Option<f32>;
    fn occludes<I: FnMut(usize, f32, f32) -> bool>(&self, ray: Ray, t_min: f32, t_max: f32, occludes: I) -> bool;
    fn depth_test<I: Fn(usize, f32, f32) -> Option<(f32, u32)>>(&self, ray: Ray, t_min: f32, t_max: f32, i: I) -> (f32, u32);
    fn traverse4<I: FnMut(usize, &mut RayPacket4)>(&self, packet: &mut RayPacket4, i: I);
}
