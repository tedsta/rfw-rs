use crate::math::*;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone)]
pub struct AABB {
    pub min: Vec3,
    pub left_first: i32,
    pub max: Vec3,
    pub count: i32,
}

impl Display for AABB {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "(min: ({}, {}, {}), left_first: {}, max: ({}, {}, {}), count: {})", self.min.x, self.min.y, self.min.z, self.left_first, self.max.x, self.max.y, self.max.z, self.count)
    }
}

impl AABB {
    pub fn new() -> AABB {
        AABB {
            min: vec3(1e34, 1e34, 1e34),
            left_first: -1,
            max: vec3(-1e34, -1e34, -1e34),
            count: -1,
        }
    }

    pub fn intersect(&self, origin: Vec3, dir_inverse: Vec3, t: f32) -> Option<(f32, f32)> {
        let t1 = (self.min.x - origin.x) * dir_inverse.x;
        let t2 = (self.max.x - origin.x) * dir_inverse.x;

        let t_min = t1.min(t2);
        let t_max = t1.max(t2);

        let t1 = (self.min.y - origin.y) * dir_inverse.y;
        let t2 = (self.max.y - origin.y) * dir_inverse.y;

        let t_min = t_min.max(t1.min(t2));
        let t_max = t_max.min(t1.max(t2));

        let t1 = (self.min.z - origin.z) * dir_inverse.z;
        let t2 = (self.max.z - origin.z) * dir_inverse.z;

        let t_min = t_min.max(t1.min(t2));
        let t_max = t_max.min(t1.max(t2));

        if t_max > t_min && t_min < t {
            return Some((t_min, t_max));
        }

        None
    }

    pub fn grow(&mut self, pos: Vec3) {
        self.min = min(self.min, pos) as Vec3;
        self.max = max(self.max, pos) as Vec3;
    }

    pub fn grow_bb(&mut self, aabb: &AABB) {
        self.min = min(self.min, aabb.min) as Vec3;
        self.max = max(self.max, aabb.max) as Vec3;
    }

    pub fn offset_by(&mut self, delta: f32) {
        self.min = self.min - vec3(delta, delta, delta);
        self.max = self.max + vec3(delta, delta, delta);
    }

    pub fn union_of(&self, bb: &AABB) -> AABB {
        AABB {
            min: min(self.min, bb.min) as Vec3,
            left_first: -1,
            max: max(self.max, bb.max) as Vec3,
            count: -1,
        }
    }

    pub fn intersection(&self, bb: &AABB) -> AABB {
        AABB {
            min: max(self.min, bb.min) as Vec3,
            left_first: -1,
            max: min(self.max, bb.max) as Vec3,
            count: -1,
        }
    }

    pub fn volume(&self) -> f32 {
        let length = self.max - self.min;
        return length.x * length.y * length.z;
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn area(&self) -> f32 {
        let e = self.max - self.min;
        let value: f32 = e.x * e.y + e.x * e.z + e.y * e.z;

        0.0_f32.max(value)
    }

    pub fn lengths(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn longest_axis(&self) -> u32 {
        let mut a: u32 = 0;
        if self.extend(1) > self.extend(0) {
            a = 1;
        }
        if self.extend(2) > self.extend(a) {
            a = 2
        }
        a
    }

    pub fn extend(&self, axis: u32) -> f32 {
        match axis {
            0 => self.max.x - self.min.x,
            1 => self.max.y - self.min.y,
            2 => self.max.z - self.min.z,
            _ => panic!(format!("Invalid axis: {}", axis))
        }
    }
}