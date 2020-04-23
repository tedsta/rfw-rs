use glam::*;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

use crate::RayPacket4;

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct AABB {
    pub min: [f32; 3],
    pub left_first: i32,
    pub max: [f32; 3],
    pub count: i32,
}

pub trait Bounds {
    fn bounds(&self) -> AABB;
}

impl Display for AABB {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let min = Vec3::from(self.min);
        let max = Vec3::from(self.max);

        write!(
            f,
            "(min: ({}, {}, {}), left_first: {}, max: ({}, {}, {}), count: {})",
            min.x(),
            min.y(),
            min.z(),
            self.left_first,
            max.x(),
            max.y(),
            max.z(),
            self.count
        )
    }
}

#[allow(dead_code)]
impl AABB {
    pub const DEFAULT_INF: f32 = 1e34;

    pub fn new() -> AABB {
        AABB {
            min: [Self::DEFAULT_INF; 3],
            left_first: -1,
            max: [-Self::DEFAULT_INF; 3],
            count: -1,
        }
    }

    pub fn intersect(&self, origin: Vec3, dir_inverse: Vec3, t: f32) -> Option<(f32, f32)> {
        let (min, max) = self.points();

        let t1 = (min - origin) * dir_inverse;
        let t2 = (max - origin) * dir_inverse;

        let t_min: Vec3 = t1.min(t2);
        let t_max: Vec3 = t1.max(t2);

        let t_min = t_min.max_element();
        let t_max = t_max.min_element();

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
        let (org_x, org_y, org_z) = packet.origin_xyz();

        let t1_x = (Vec4::splat(self.min[0]) - org_x) * inv_dir_x;
        let t1_y = (Vec4::splat(self.min[1]) - org_y) * inv_dir_y;
        let t1_z = (Vec4::splat(self.min[2]) - org_z) * inv_dir_z;

        let t2_x = (Vec4::splat(self.max[0]) - org_x) * inv_dir_x;
        let t2_y = (Vec4::splat(self.max[1]) - org_y) * inv_dir_y;
        let t2_z = (Vec4::splat(self.max[2]) - org_z) * inv_dir_z;

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

    pub fn grow(&mut self, pos: Vec3) {
        let (min, max) = self.points();
        let min = min.min(pos);
        let max = max.max(pos);

        for i in 0..3 {
            self.min[i] = min[i];
            self.max[i] = max[i];
        }
    }

    pub fn grow_bb(&mut self, aabb: &AABB) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(aabb.min[i]);
            self.max[i] = self.max[i].max(aabb.max[i]);
        }
    }

    pub fn offset_by(&mut self, delta: f32) {
        let delta = Vec3::from([delta; 3]);

        let (min, max) = self.points();
        let min = min - delta;
        let max = max + delta;

        for i in 0..3 {
            self.min[i] = min[i];
            self.max[i] = max[i];
        }
    }

    pub fn union_of(&self, bb: &AABB) -> AABB {
        let (min, max) = self.points();
        let (b_min, b_max) = bb.points();

        let new_min = Vec3::from(min).min(Vec3::from(b_min));
        let new_max = Vec3::from(max).max(Vec3::from(b_max));

        AABB {
            min: new_min.into(),
            left_first: -1,
            max: new_max.into(),
            count: -1,
        }
    }

    pub fn intersection(&self, bb: &AABB) -> AABB {
        let (min, max) = self.points();
        let (b_min, b_max) = bb.points();

        let new_min = Vec3::from(min).max(Vec3::from(b_min));
        let new_max = Vec3::from(max).min(Vec3::from(b_max));

        AABB {
            min: new_min.into(),
            left_first: -1,
            max: new_max.into(),
            count: -1,
        }
    }

    pub fn volume(&self) -> f32 {
        let length = Vec3::from(self.max) - Vec3::from(self.min);
        return length.x() * length.y() * length.z();
    }

    pub fn center(&self) -> Vec3 {
        (Vec3::from(self.min) + Vec3::from(self.max)) * 0.5
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

    pub fn transformed(&self, transform: Mat4) -> AABB {
        let p1 = transform * Vec3::new(self.min[0], self.min[1], self.min[2]).extend(1.0);
        let p5 = transform * Vec3::new(self.max[0], self.max[1], self.max[2]).extend(1.0);
        let p2 = transform * Vec3::new(self.max[0], self.min[1], self.min[2]).extend(1.0);
        let p3 = transform * Vec3::new(self.min[0], self.max[1], self.max[2]).extend(1.0);
        let p4 = transform * Vec3::new(self.min[0], self.min[1], self.max[2]).extend(1.0);
        let p6 = transform * Vec3::new(self.max[0], self.max[1], self.min[2]).extend(1.0);
        let p7 = transform * Vec3::new(self.min[0], self.max[1], self.min[2]).extend(1.0);
        let p8 = transform * Vec3::new(self.max[0], self.min[1], self.max[2]).extend(1.0);

        let mut transformed = AABB::new();
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

    pub fn points(&self) -> (Vec3, Vec3) {
        (
            Vec3::from(unsafe { core::arch::x86_64::_mm_load_ps(self.min.as_ptr()) }),
            Vec3::from(unsafe { core::arch::x86_64::_mm_load_ps(self.max.as_ptr()) }),
        )
    }

    pub fn transform(&mut self, transform: Mat4) {
        let transformed = self.transformed(transform);
        *self = AABB {
            min: transformed.min,
            max: transformed.max,
            ..(*self).clone()
        }
    }

    pub fn ensure_non_zero(&mut self) {
        for i in 0..3 {
            if self.max[i] - self.min[i] < 0.001 {
                self.max[i] = self.min[i] + 0.005;
            }
        }
    }

    pub fn is_valid(&self) -> bool {
        let (min, max) = self.points();
        max.cmpgt(min).all()
    }

    pub fn is_empty(&self) -> bool {
        let (min, max) = self.points();

        let mask1 = min.cmpeq(Vec3::splat(-Self::DEFAULT_INF));
        let mask2 = max.cmpeq(Vec3::splat(Self::DEFAULT_INF));

        (mask1 & mask2).all()
    }

    pub fn overlap(b1: &AABB, b2: &AABB) -> AABB {
        assert!(b1.is_valid() || b1.is_empty());
        assert!(b2.is_valid() || b2.is_empty());

        let (min1, max1) = b1.points();
        let (min2, max2) = b2.points();

        let min = min1.max(min2);
        let max = max1.min(max2);

        let aabb: AABB = (min, max).into();

        match aabb.is_valid() {
            true => aabb,
            _ => AABB::new()
        }
    }
}

impl From<(Vec3, Vec3)> for AABB {
    fn from((min, max): (Vec3, Vec3)) -> Self {
        AABB {
            min: min.into(),
            left_first: -1,
            max: max.into(),
            count: -1,
        }
    }
}

impl Into<(Vec3, Vec3)> for AABB {
    fn into(self) -> (Vec3, Vec3) {
        self.points()
    }
}