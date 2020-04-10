use glam::*;
use serde::{Serialize, Deserialize};
use std::fmt::{Display, Formatter};

use crate::RayPacket4;

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

pub trait Bounds {
    fn bounds(&self) -> Aabb;
}

impl Display for Aabb {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let min = Vec3::from(self.min);
        let max = Vec3::from(self.max);

        write!(
            f,
            "(min: ({}, {}, {}),  max: ({}, {}, {}))",
            min.x(),
            min.y(),
            min.z(),
            max.x(),
            max.y(),
            max.z(),
        )
    }
}

#[allow(dead_code)]
impl Aabb {
    pub fn new() -> Aabb {
        Aabb {
            min: [1e34; 3],
            max: [-1e34; 3],
        }
    }

    pub fn intersect(&self, origin: Vec3, dir_inverse: Vec3, t: f32) -> Option<(f32, f32)> {
        let min = Vec3::from(self.min);
        let max = Vec3::from(self.max);

        let t1 = (min - origin) * dir_inverse;
        let t2 = (max - origin) * dir_inverse;

        let t_min = t1.min(t2);
        let t_max = t1.max(t2);

        let t_min = t_min.x().max(t_min.y().max(t_min.z()));
        let t_max = t_max.x().min(t_max.y().min(t_max.z()));

        if t_max > t_min && t_min < t {
            return Some((t_min, t_max));
        }

        None
    }

    pub fn intersect4(
        &self,
        packet: &RayPacket4,
        inv_dir_x: Vec4,
        inv_dir_y: Vec4,
        inv_dir_z: Vec4,
    ) -> Option<[f32; 4]> {
        let org_x: Vec4 = packet.origin_x.into();
        let org_y: Vec4 = packet.origin_y.into();
        let org_z: Vec4 = packet.origin_z.into();

        let t1_x = (Vec4::from([self.min[0]; 4]) - org_x) * inv_dir_x;
        let t1_y = (Vec4::from([self.min[1]; 4]) - org_y) * inv_dir_y;
        let t1_z = (Vec4::from([self.min[2]; 4]) - org_z) * inv_dir_z;

        let t2_x = (Vec4::from([self.max[0]; 4]) - org_x) * inv_dir_x;
        let t2_y = (Vec4::from([self.max[1]; 4]) - org_y) * inv_dir_y;
        let t2_z = (Vec4::from([self.max[2]; 4]) - org_z) * inv_dir_z;

        let t_min_x = t1_x.min(t2_x);
        let t_min_y = t1_y.min(t2_y);
        let t_min_z = t1_z.min(t2_z);

        let t_max_x = t1_x.max(t2_x);
        let t_max_y = t1_y.max(t2_y);
        let t_max_z = t1_z.max(t2_z);

        let t_min = t_min_x.max(t_min_y.max(t_min_z));
        let t_max = t_max_x.min(t_max_y.min(t_max_z));

        let mask = t_max.cmpgt(t_min) & t_min.cmplt(Vec4::from(packet.t));
        if mask.any() {
            Some(t_min.into())
        } else {
            None
        }
    }

    pub fn union(&mut self, pos: Vec3) {
        let (min, max) = self.points();

        let min = min.min(pos);
        let max = max.max(pos);

        self.min = min.into();
        self.max = max.into();
    }

    pub fn grow(&mut self, pos: Vec3) {
        let (min, max) = self.points();

        let min = min.min(pos);
        let max = max.max(pos);

        self.min = min.into();
        self.max = max.into();
    }

    pub fn grow_bb(&mut self, aabb: &Aabb) {
        let (min, max) = self.points();
        let (b_min, b_max) = aabb.points();
        self.min = min.min(b_min).into();
        self.max = max.max(b_max).into();
    }

    pub fn offset_by(&mut self, delta: f32) {
        let delta = Vec3::from([delta; 3]);
        let (min, max) = self.points();
        self.min = (min - delta).into();
        self.max = (max + delta).into();
    }

    pub fn union_of(&self, bb: &Self) -> Self {
        let (min, max) = self.points();
        let (b_min, b_max) = bb.points();
        Self { min: min.min(b_min).into(), max: max.max(b_max).into() }
    }

    pub fn intersection(&self, bb: &Self) -> Self {
        let (min, max) = self.points();
        let (b_min, b_max) = bb.points();
        Self { min: min.max(b_min).into(), max: max.min(b_max).into() }
    }

    pub fn volume(&self) -> f32 {
        let length = Vec3::from(self.max) - Vec3::from(self.min);
        return length.x() * length.y() * length.z();
    }

    pub fn center(&self) -> Vec3 {
        let (min, max) = self.points();
        (min + max) * 0.5
    }

    pub fn offset(&self, p: Vec3) -> Vec3 {
        let (min, max) = self.points();
        let o = p - min;
        o / max.cmpgt(min).select(max - min, Vec3::one())
    }

    pub fn area(&self) -> f32 {
        let e = Vec3::from(self.max) - Vec3::from(self.min);
        let value: f32 = e.x() * e.y() + e.x() * e.z() + e.y() * e.z();

        0.0_f32.max(value)
    }

    pub fn lengths(&self) -> Vec3 {
        Vec3::from(self.max) - Vec3::from(self.min)
    }

    pub fn longest_axis(&self) -> usize {
        let mut a: usize = 0;
        if self.extend(1) > self.extend(0) {
            a = 1;
        }
        if self.extend(2) > self.extend(a) {
            a = 2
        }
        a
    }

    pub fn all_corners(&self) -> [Vec3; 8] {
        let lengths: Vec3 = self.lengths();

        let x_l = Vec3::splat(lengths.x());
        let y_l = Vec3::splat(lengths.y());
        let z_l = Vec3::splat(lengths.z());

        let min = Vec3::from(self.min);
        let max = Vec3::from(self.max);

        [
            min,
            max,
            min + x_l,
            min + y_l,
            min + z_l,
            min + x_l + y_l,
            min + x_l + z_l,
            min + y_l + z_l,
        ]
    }

    pub fn extend(&self, axis: usize) -> f32 {
        self.max[axis] - self.min[axis]
    }

    pub fn transformed(&self, transform: Mat4) -> Aabb {
        let p1 = transform * Vec3::new(self.min[0], self.min[1], self.min[2]).extend(1.0);
        let p5 = transform * Vec3::new(self.max[0], self.max[1], self.max[2]).extend(1.0);
        let p2 = transform * Vec3::new(self.max[0], self.min[1], self.min[2]).extend(1.0);
        let p3 = transform * Vec3::new(self.min[0], self.max[1], self.max[2]).extend(1.0);
        let p4 = transform * Vec3::new(self.min[0], self.min[1], self.max[2]).extend(1.0);
        let p6 = transform * Vec3::new(self.max[0], self.max[1], self.min[2]).extend(1.0);
        let p7 = transform * Vec3::new(self.min[0], self.max[1], self.min[2]).extend(1.0);
        let p8 = transform * Vec3::new(self.max[0], self.min[1], self.max[2]).extend(1.0);

        let mut transformed = Aabb::new();
        transformed.grow(p1.truncate());
        transformed.grow(p2.truncate());
        transformed.grow(p3.truncate());
        transformed.grow(p4.truncate());
        transformed.grow(p5.truncate());
        transformed.grow(p6.truncate());
        transformed.grow(p7.truncate());
        transformed.grow(p8.truncate());

        transformed.offset_by(1e-4);

        transformed
    }

    pub fn transform(&mut self, transform: Mat4) {
        let transformed = self.transformed(transform);
        *self = Self {
            min: transformed.min,
            max: transformed.max,
        }
    }

    pub fn points(&self) -> (Vec3, Vec3) {
        (self.min.into(), self.max.into())
    }
}

impl From<(Vec3, Vec3)> for Aabb {
    fn from(v: (Vec3, Vec3)) -> Self {
        Aabb {
            min: v.0.into(),
            max: v.1.into(),
        }
    }
}

impl Into<(Vec3, Vec3)> for Aabb {
    fn into(self) -> (Vec3, Vec3) {
        self.points()
    }
}