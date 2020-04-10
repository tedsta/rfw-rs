use glam::*;
use std::convert::Into;
use bvh::Aabb;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum FrustrumResult {
    Outside,
    Intersect,
    Inside,
}

#[derive(Copy, Clone)]
pub struct FrustrumPlane {
    pub normal: Vec3,
    pub d: f32,
}

impl FrustrumPlane {
    pub fn from_coefficients(a: f32, b: f32, c: f32, d: f32) -> FrustrumPlane {
        let normal = Vec3::from([a, b, c]);
        let length = normal.length();
        let normal = normal / length;

        Self {
            normal: normal.into(),
            d: d / length,
        }
    }

    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3) -> FrustrumPlane {
        let aux1 = v0 - v1;
        let aux2 = v2 - v1;

        let normal = (aux2 * aux1).normalize();

        let point = v2;
        let d = -(normal.dot(point));

        FrustrumPlane { normal, d }
    }

    pub fn set_3_points(&mut self, v0: Vec3, v1: Vec3, v2: Vec3) {
        let aux1 = v0 - v1;
        let aux2 = v2 - v1;

        let normal = (aux2 * aux1).normalize();

        let point = v2;
        let d = -(normal.dot(point));

        self.normal = normal;
        self.d = d;
    }

    pub fn set_normal_and_point(&mut self, normal: Vec3, point: Vec3) {
        self.normal = normal;
        self.d = -(normal.dot(point));
    }

    pub fn set_coefficients(&mut self, a: f32, b: f32, c: f32, d: f32) {
        let normal = Vec3::from([a, b, c]);
        let length = normal.length();
        let normal = normal / length;
        self.normal = normal;
        self.d = d / length;
    }

    pub fn distance(&self, p: Vec3) -> f32 {
        self.d + self.normal.dot(p)
    }
}

impl From<(Vec3, Vec3, Vec3)> for FrustrumPlane {
    fn from(vecs: (Vec3, Vec3, Vec3)) -> Self {
        FrustrumPlane::new(vecs.0, vecs.1, vecs.2)
    }
}

impl From<[Vec3; 3]> for FrustrumPlane {
    fn from(vecs: [Vec3; 3]) -> Self {
        FrustrumPlane::new(vecs[0], vecs[1], vecs[2])
    }
}

impl From<(f32, f32, f32, f32)> for FrustrumPlane {
    fn from(coeffs: (f32, f32, f32, f32)) -> Self {
        let (a, b, c, d) = coeffs;
        Self::from_coefficients(a, b, c, d)
    }
}

impl From<[f32; 4]> for FrustrumPlane {
    fn from(coeffs: [f32; 4]) -> Self {
        let [a, b, c, d] = coeffs;
        Self::from_coefficients(a, b, c, d)
    }
}

enum FrustrumSide {
    Top = 0,
    Bottom = 1,
    Left = 2,
    Right = 3,
    NearPlane = 4,
    FarPlane = 5,
}

impl Into<usize> for FrustrumSide {
    fn into(self) -> usize {
        self as usize
    }
}

pub struct FrustrumG {
    pub planes: [FrustrumPlane; 6],
}

impl FrustrumG {
    pub fn from_matrix(matrix: Mat4) -> FrustrumG {
        let ma = matrix.to_cols_array();

        let a = ma[3];
        let b = ma[7];
        let c = ma[11];
        let d = ma[15];

        // Left
        let plane0: FrustrumPlane = (a + ma[0], b + ma[4], c + ma[8], d + ma[12]).into();
        // Right
        let plane1: FrustrumPlane = (a - ma[0], b - ma[4], c - ma[8], d - ma[12]).into();
        // Top
        let plane2: FrustrumPlane = (a - ma[1], b - ma[5], c - ma[9], d - ma[13]).into();
        // Bottom
        let plane3: FrustrumPlane = (a + ma[1], b + ma[5], c + ma[9], d + ma[13]).into();
        let plane4: FrustrumPlane = (a + ma[2], b + ma[6], c + ma[10], d + ma[14]).into();
        let plane5: FrustrumPlane = (a - ma[2], b - ma[6], c - ma[10], d - ma[14]).into();

        FrustrumG {
            planes: [plane0, plane1, plane2, plane3, plane4, plane5],
        }
    }

    pub fn new(camera: &crate::camera::Camera) -> FrustrumG {
        let matrix = camera.get_rh_matrix();

        Self::from_matrix(matrix)
    }

    pub fn point_in_frustrum(&self, p: Vec3) -> FrustrumResult {
        for plane in &self.planes {
            if plane.distance(p) < 0.0 {
                return FrustrumResult::Outside;
            }
        }

        FrustrumResult::Inside
    }

    pub fn sphere_in_frustrum(&self, p: Vec3, radius: f32) -> FrustrumResult {
        let mut result = FrustrumResult::Inside;
        for plane in &self.planes {
            let distance = plane.distance(p);
            if distance < -radius {
                return FrustrumResult::Outside;
            } else {
                result = FrustrumResult::Intersect;
            }
        }
        result
    }

    pub fn aabb_in_frustrum(&self, b: &Aabb) -> FrustrumResult {
        let mut result = FrustrumResult::Outside;

        for plane in &self.planes {
            let mut min = [0.0; 3];
            let mut max = [0.0; 3];

            if plane.normal.x() > 0.0 {
                min[0] = b.min[0];
                max[0] = b.max[0];
            } else {
                min[0] = b.max[0];
                max[0] = b.min[0];
            }

            if plane.normal.y() > 0.0 {
                min[1] = b.min[1];
                max[1] = b.max[1];
            } else {
                min[1] = b.max[1];
                max[1] = b.min[1];
            }

            if plane.normal.z() > 0.0 {
                min[2] = b.min[2];
                max[2] = b.max[2];
            } else {
                min[2] = b.max[2];
                max[2] = b.min[2];
            }

            if plane.distance(min.into()) >= 0.0 || plane.distance(max.into()) >= 0.0 {
                result = FrustrumResult::Intersect;
            } else {
                return FrustrumResult::Outside;
            }
        }

        result
    }
}

impl From<&crate::camera::Camera> for FrustrumG {
    fn from(camera: &crate::camera::Camera) -> Self {
        Self::new(camera)
    }
}

impl From<Mat4> for FrustrumG {
    fn from(matrix: Mat4) -> Self {
        Self::from_matrix(matrix)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn frustrum_works() {
        use crate::camera::*;
        use bvh::Aabb;

        let camera = Camera::zero();

        let frustrum: FrustrumG = FrustrumG::from_matrix(camera.get_rh_matrix());

        let point_behind = Vec3::new(0.0, 0.0, -1.0);
        let point_in_front = Vec3::new(0.0, 0.0, 1.0);
        let aabb_in_front = Aabb {
            min: Vec3::new(0.2, 0.2, 5.0).into(),
            max: Vec3::new(0.2, 0.2, 5.0).into(),
        };

        let aabb_in_back = Aabb {
            min: Vec3::new(-1.0, 0.0, -2.0).into(),
            max: Vec3::new(1.0, 0.0, -2.0).into(),
        };

        let aabb_half = Aabb {
            min: Vec3::new(-5.0, 0.0, 2.0).into(),
            max: Vec3::new(0.0, 0.0, 2.0).into(),
        };

        assert_eq!(
            FrustrumResult::Outside,
            frustrum.point_in_frustrum(point_behind),
        );
        assert_eq!(
            FrustrumResult::Inside,
            frustrum.point_in_frustrum(point_in_front),
        );
        assert_eq!(
            FrustrumResult::Intersect,
            frustrum.aabb_in_frustrum(&aabb_in_front)
        );
        assert_eq!(
            FrustrumResult::Outside,
            frustrum.aabb_in_frustrum(&aabb_in_back),
        );
        assert_eq!(
            FrustrumResult::Intersect,
            frustrum.aabb_in_frustrum(&aabb_half),
        );
    }
}
