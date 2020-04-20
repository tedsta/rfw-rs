use glam::*;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct AreaLight {
    position: [f32; 3],
    energy: f32,
    normal: [f32; 3],
    tri_idx: i32,
    vertex0: [f32; 3],
    inst_idx: i32,
    vertex1: [f32; 3],
    vertex2: [f32; 3],
}

impl AreaLight {
    pub fn new(
        pos: Vec3,
        radiance: Vec3,
        normal: Vec3,
        tri_id: i32,
        inst_id: i32,
        vertex0: Vec3,
        vertex1: Vec3,
        vertex2: Vec3,
    ) -> AreaLight {
        let energy = radiance.length();
        Self {
            position: pos.into(),
            energy,
            normal: normal.into(),
            tri_idx: tri_id,
            vertex0: vertex0.into(),
            inst_idx: inst_id,
            vertex1: vertex1.into(),
            vertex2: vertex2.into(),
        }
    }
} // 72 Bytes

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct PointLight {
    position: [f32; 3],
    energy: f32,
    radiance: [f32; 3],
    _dummy: i32,
}

impl PointLight {
    pub fn new(
        position: Vec3,
        radiance: Vec3,
    ) -> PointLight {
        Self {
            position: position.into(),
            energy: radiance.length(),
            radiance: radiance.into(),
            _dummy: 0,
        }
    }
} // 32 Bytes

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct SpotLight
{
    position: [f32; 3],
    cos_inner: f32,
    radiance: [f32; 3],
    cos_outer: f32,
    direction: [f32; 3],
    energy: f32,
} // 48 Bytes

impl SpotLight {
    pub fn new(
        position: Vec3,
        direction: Vec3,
        inner_angle: f32,
        outer_angle: f32,
        radiance: Vec3,
    ) -> SpotLight {
        let inner_angle = inner_angle.to_radians();
        let outer_angle = outer_angle.to_radians();

        Self {
            position: position.into(),
            cos_inner: inner_angle.cos(),
            radiance: radiance.into(),
            cos_outer: outer_angle.cos(),
            direction: direction.into(),
            energy: radiance.length(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct DirectionalLight
{
    direction: [f32; 3],
    energy: f32,
    radiance: [f32; 3],
    _dummy: i32,
} // 32 Bytes

impl DirectionalLight {
    pub fn new(
        direction: Vec3,
        radiance: Vec3,
    ) -> DirectionalLight {
        Self {
            direction: direction.into(),
            energy: radiance.length(),
            radiance: radiance.into(),
            _dummy: 0,
        }
    }
}