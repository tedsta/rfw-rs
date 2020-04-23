use glam::*;
use rayon::prelude::*;

use crate::objects::*;
use crate::scene::{PrimID, USE_MBVH};
use bvh::{Bounds, Ray, RayPacket4, AABB, BVH, MBVH};
use serde::{Deserialize, Serialize};

pub trait ToMesh {
    fn into_mesh(self) -> Mesh;
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct VertexData {
    pub vertex: [f32; 4],
    pub normal: [f32; 3],
    pub mat_id: u32,
    pub uv: [f32; 2],
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct VertexMesh {
    pub first: u32,
    pub last: u32,
    pub mat_id: u32,
    pub bounds: AABB,
}

pub struct VertexBuffer {
    pub count: usize,
    pub size_in_bytes: usize,
    pub buffer: wgpu::Buffer,
    pub bounds: AABB,
    pub meshes: Vec<VertexMesh>,
}

impl VertexData {
    pub fn zero() -> VertexData {
        VertexData {
            vertex: [0.0, 0.0, 0.0, 1.0],
            normal: [0.0; 3],
            mat_id: 0,
            uv: [0.0; 2],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mesh {
    pub triangles: Vec<RTTriangle>,
    pub vertices: Vec<VertexData>,
    pub materials: Vec<u32>,
    pub meshes: Vec<VertexMesh>,
    pub bounds: AABB,
    pub bvh: BVH,
    pub mbvh: MBVH,
}

impl Mesh {
    pub fn new(vertices: &[Vec3], normals: &[Vec3], uvs: &[Vec2], material_ids: &[u32]) -> Mesh {
        assert_eq!(vertices.len(), normals.len());
        assert_eq!(vertices.len(), uvs.len());
        assert_eq!(uvs.len(), material_ids.len() * 3);
        assert_eq!(vertices.len() % 3, 0);

        let mut bounds = AABB::new();
        let mut vertex_data = vec![VertexData::zero(); vertices.len()];

        for vertex in vertices {
            bounds.grow(*vertex);
        }

        vertex_data.par_iter_mut().enumerate().for_each(|(i, v)| {
            let vertex: [f32; 3] = vertices[i].into();
            let vertex = [vertex[0], vertex[1], vertex[2], 1.0];
            let normal = normals[i].into();
            *v = VertexData {
                vertex,
                normal,
                mat_id: material_ids[i / 3],
                uv: uvs[i].into(),
            };
        });

        let mut last_id = material_ids[0];
        let mut start = 0;
        let mut range = 0;
        let mut meshes: Vec<VertexMesh> = Vec::new();
        let mut v_bounds = AABB::new();

        for i in 0..material_ids.len() {
            range += 1;
            for j in 0..3 {
                v_bounds.grow(vertices[i * 3 + j]);
            }

            if last_id != material_ids[i] {
                bounds = AABB::new();
                meshes.push(VertexMesh {
                    first: start * 3,
                    last: (start + range) * 3,
                    mat_id: last_id,
                    bounds: v_bounds.clone(),
                });

                last_id = material_ids[i];
                start = i as u32;
                range = 1;
            }
        }

        let mut triangles = vec![RTTriangle::zero(); vertices.len() / 3];
        triangles.iter_mut().enumerate().for_each(|(i, triangle)| {
            let i0 = i * 3;
            let i1 = i0 + 1;
            let i2 = i0 + 2;

            let vertex0 = unsafe { *vertices.get_unchecked(i0) };
            let vertex1 = unsafe { *vertices.get_unchecked(i1) };
            let vertex2 = unsafe { *vertices.get_unchecked(i2) };

            let n0 = unsafe { *normals.get_unchecked(i0) };
            let n1 = unsafe { *normals.get_unchecked(i1) };
            let n2 = unsafe { *normals.get_unchecked(i2) };

            let uv0 = unsafe { *uvs.get_unchecked(i0) };
            let uv1 = unsafe { *uvs.get_unchecked(i1) };
            let uv2 = unsafe { *uvs.get_unchecked(i2) };

            let normal = RTTriangle::normal(vertex0, vertex1, vertex2);

            *triangle = RTTriangle {
                vertex0: vertex0.into(),
                u0: uv0.x(),
                vertex1: vertex1.into(),
                u1: uv1.x(),
                vertex2: vertex2.into(),
                u2: uv2.x(),
                normal: normal.into(),
                v0: uv0.y(),
                n0: n0.into(),
                v1: uv1.y(),
                n1: n1.into(),
                v2: uv2.y(),
                n2: n2.into(),
                id: i as i32,
                light_id: -1,
            };
        });

        let aabbs: Vec<AABB> = triangles.iter().map(|t| t.bounds()).collect();
        let bvh = BVH::construct_sbvh(aabbs.as_slice(), triangles.as_slice());
        // let bvh = BVH::construct(aabbs.as_slice());
        let mbvh = MBVH::construct(&bvh);

        Mesh {
            triangles,
            vertices: vertex_data,
            materials: Vec::from(material_ids),
            meshes,
            bounds,
            bvh,
            mbvh,
        }
    }

    pub fn scale(&mut self, scaling: f32) -> Self {
        let mut new_self = self.clone();

        let scaling = Mat4::from_scale(Vec3::splat(scaling));
        new_self.triangles.par_iter_mut().for_each(|t| {
            let vertex0 = scaling * Vec4::new(t.vertex0[0], t.vertex0[1], t.vertex0[2], 1.0);
            let vertex1 = scaling * Vec4::new(t.vertex1[0], t.vertex1[1], t.vertex1[2], 1.0);
            let vertex2 = scaling * Vec4::new(t.vertex2[0], t.vertex2[1], t.vertex2[2], 1.0);

            t.vertex0 = vertex0.truncate().into();
            t.vertex1 = vertex1.truncate().into();
            t.vertex2 = vertex2.truncate().into();
        });

        new_self.vertices.iter_mut().for_each(|v| {
            v.vertex = (scaling * Vec4::new(v.vertex[0], v.vertex[1], v.vertex[2], 1.0)).into();
        });

        let aabbs: Vec<AABB> = new_self.triangles.iter().map(|t| t.bounds()).collect();

        new_self.bvh = BVH::construct(aabbs.as_slice());
        new_self.mbvh = MBVH::construct(&new_self.bvh);

        new_self
    }

    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    pub fn empty() -> Mesh {
        Mesh {
            triangles: Vec::new(),
            vertices: Vec::new(),
            materials: Vec::new(),
            meshes: Vec::new(),
            bounds: AABB::new(),
            bvh: BVH::empty(),
            mbvh: MBVH::empty(),
        }
    }

    pub fn buffer_size(&self) -> usize {
        self.vertices.len() * std::mem::size_of::<VertexData>()
    }

    pub fn as_slice(&self) -> &[VertexData] {
        self.vertices.as_slice()
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.vertices.as_ptr() as *const u8, self.buffer_size())
        }
    }
}

impl Intersect for Mesh {
    fn occludes(&self, ray: Ray, t_min: f32, t_max: f32) -> bool {
        let (origin, direction) = ray.into();

        let intersection_test = |i, t_min, t_max| {
            let triangle: &RTTriangle = unsafe { self.triangles.get_unchecked(i) };
            triangle.occludes(ray, t_min, t_max)
        };

        unsafe {
            match USE_MBVH {
                true => self.mbvh.occludes(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection_test,
                ),
                _ => self.bvh.occludes(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection_test,
                ),
            }
        }
    }

    fn intersect(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let (origin, direction) = ray.into();

        let intersection_test = |i, t_min, t_max| {
            let triangle: &RTTriangle = unsafe { self.triangles.get_unchecked(i) };
            if let Some(mut hit) = triangle.intersect(ray, t_min, t_max) {
                hit.mat_id = self.materials[i];
                return Some((hit.t, hit));
            }
            None
        };

        unsafe {
            match USE_MBVH {
                true => self.mbvh.traverse(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection_test,
                ),
                _ => self.bvh.traverse(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection_test,
                ),
            }
        }
    }

    fn intersect_t(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<f32> {
        let (origin, direction) = ray.into();

        let intersection_test = |i, t_min, t_max| {
            let triangle: &RTTriangle = unsafe { self.triangles.get_unchecked(i) };
            if let Some(t) = triangle.intersect_t(ray, t_min, t_max) {
                return Some(t);
            }
            None
        };

        unsafe {
            match USE_MBVH {
                true => self.mbvh.traverse_t(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection_test,
                ),
                _ => self.bvh.traverse_t(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection_test,
                ),
            }
        }
    }

    fn depth_test(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<(f32, u32)> {
        let (origin, direction) = ray.into();

        let intersection_test = |i, t_min, t_max| -> Option<(f32, u32)> {
            let triangle: &RTTriangle = unsafe { self.triangles.get_unchecked(i) };
            triangle.depth_test(ray, t_min, t_max)
        };

        let hit = unsafe {
            match USE_MBVH {
                true => self.mbvh.depth_test(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection_test,
                ),
                _ => self.bvh.depth_test(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection_test,
                ),
            }
        };

        Some(hit)
    }

    fn intersect4(&self, packet: &mut RayPacket4, t_min: &[f32; 4]) -> Option<[PrimID; 4]> {
        let mut prim_id = [-1 as PrimID; 4];
        let mut valid = false;
        let intersection_test = |i: usize, packet: &mut RayPacket4| {
            let triangle: &RTTriangle = unsafe { self.triangles.get_unchecked(i) };
            if let Some(hit) = triangle.intersect4(packet, t_min) {
                valid = true;
                for i in 0..4 {
                    if hit[i] >= 0 {
                        prim_id[i] = hit[i];
                    }
                }
            }
        };

        unsafe {
            match USE_MBVH {
                true => self.mbvh.traverse4(packet, intersection_test),
                _ => self.bvh.traverse4(packet, intersection_test),
            }
        };

        if valid {
            Some(prim_id)
        } else {
            None
        }
    }

    fn get_hit_record(&self, ray: Ray, t: f32, hit_data: u32) -> HitRecord {
        self.triangles[hit_data as usize].get_hit_record(ray, t, hit_data)
    }
}

impl Bounds for Mesh {
    fn bounds(&self) -> AABB {
        self.bounds.clone()
    }
}

impl<'a> SerializableObject<'a, Mesh> for Mesh {
    fn serialize<S: AsRef<std::path::Path>>(
        &self,
        path: S,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;
        let encoded: Vec<u8> = bincode::serialize(self)?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(encoded.as_ref())?;
        Ok(())
    }

    fn deserialize<S: AsRef<std::path::Path>>(path: S) -> Result<Mesh, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let object: Self = bincode::deserialize_from(reader)?;
        Ok(object)
    }
}
