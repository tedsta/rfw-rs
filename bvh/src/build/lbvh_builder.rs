use crate::{Aabb, BVHBuilder, RayPacket4, Ray, Bounds};
use glam::*;
use rayon::prelude::*;
use std::ops::Index;
use serde::{Serialize, Deserialize};

struct WorkDivision {
    idx: usize,
    max: usize,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
struct Node {
    pub bounds: Aabb,
    pub left: u32,
    pub right: u32,
}

const fn highest_bit() -> u32 {
    1 << ((std::mem::size_of::<u32>() as u32 * 8) - 1)
}

impl Node {
    pub fn new() -> Self {
        Self {
            bounds: Aabb::new(),
            left: 0,
            right: 0,
        }
    }

    pub fn left_leaf_index(&self) -> u32 {
        let highest_bit = highest_bit() - 1;
        self.left & highest_bit
    }

    pub fn right_leaf_index(&self) -> u32 {
        let highest_bit = highest_bit() - 1;
        self.right & highest_bit
    }

    pub fn left_is_leaf(&self) -> bool {
        let highest_bit = highest_bit();
        (self.left & highest_bit) == highest_bit
    }

    pub fn right_is_leaf(&self) -> bool {
        let highest_bit = highest_bit();
        (self.right & highest_bit) == highest_bit
    }
}

fn clz(n: u32) -> u32 {
    unsafe { core::arch::x86_64::_lzcnt_u32(n) }
}

fn ceil_div(n: i32, d: i32) -> i32 {
    (n / d) + (if (n % d) > 0 { 1 } else { 0 })
}

fn hadamard_division(a: Vec3, b: Vec3) -> Vec3 {
    a / b
}

struct LoopRange {
    begin: usize,
    end: usize,
}

impl LoopRange {
    pub fn new(div: WorkDivision, array_size: usize) -> Self {
        let chunk_size = array_size / div.max;
        let begin = chunk_size * div.idx;
        let end = if (div.idx + 1) == div.max {
            array_size
        } else {
            begin + chunk_size
        };

        Self { begin, end }
    }
}

// Represents a single entry within the curve table.
#[derive(Debug, Copy, Clone)]
struct MortonEntry {
    // The code at this point along the curve.
    pub code: u32,
    // The index to the primitive associated with this point.
    primitive: u32,
}

const MORTON_DOMAIN: usize = 1024;

fn morton_expand(mut n: u32) -> u32 {
    n = (n | (n << 16)) & 0x030000ff;
    n = (n | (n << 8)) & 0x0300f00f;
    n = (n | (n << 4)) & 0x030c30c3;
    n = (n | (n << 2)) & 0x09249249;
    n
}

fn morton_encoder(x: u32, y: u32, z: u32) -> u32 {
    (morton_expand(x) << 2) | (morton_expand(y) << 1) | (morton_expand(z) << 0)
}

#[derive(Debug, Clone)]
struct SpaceFillingCurve {
    entries: Vec<MortonEntry>,
}

impl Index<usize> for SpaceFillingCurve {
    type Output = MortonEntry;
    fn index(&self, index: usize) -> &Self::Output {
        self.entries.get(index).unwrap()
    }
}

impl SpaceFillingCurve {
    pub fn new(prims: &[Aabb], scene_bounds: &Aabb) -> Self {
        let entries = Self::morton_curve_kernel(&scene_bounds, prims);
        Self { entries }
    }

    fn morton_curve_kernel(scene_box: &Aabb, p: &[Aabb]) -> Vec<MortonEntry> {
        let m_domain = MORTON_DOMAIN as f32;
        let scene_size = scene_box.lengths();
        (0..p.len())
            .into_par_iter()
            .map(|i| {
                let i = i as usize;
                let center: Vec3 = p[i].center();

                let normalized_center =
                    hadamard_division(center - Vec3::from(scene_box.min), scene_size);

                let morton_center: Vec3 = normalized_center * m_domain;
                let (x, y, z) = morton_center.into();
                let x = x as u32;
                let y = y as u32;
                let z = z as u32;

                let code = morton_encoder(x, y, z);
                MortonEntry {
                    code,
                    primitive: i as u32,
                }
            })
            .collect::<Vec<MortonEntry>>()
    }

    pub fn sort(&mut self) {
        self.entries.sort_by(|a, b| {
            if a.code < b.code {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });
    }

    pub fn sorted(mut self) -> Self {
        self.entries.sort_by(|a, b| {
            if a.code < b.code {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        self
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

struct NodeDivision {
    pub index_a: u32,
    pub index_b: u32,
    pub split: u32,
}

impl NodeDivision {
    pub fn min(&self) -> u32 {
        self.index_a.min(self.index_b)
    }

    pub fn max(&self) -> u32 {
        self.index_a.max(self.index_b)
    }

    fn calc_delta(table: &SpaceFillingCurve, j: i32, k: i32) -> i32 {
        if k < 0 || k >= table.len() as i32 {
            return -1;
        }

        let l_code = table[j as usize].code;
        let r_code = table[k as usize].code;

        (if l_code == r_code {
            clz((j ^ k) as u32) + 32
        } else {
            clz((l_code ^ r_code) as u32)
        }) as i32
    }

    pub fn divide_node(table: &SpaceFillingCurve, node_idx: usize) -> Self {
        let i = node_idx as i32;
        let table_size = table.len() as i32;
        let d = Self::calc_delta(table, i, i + 1) - Self::calc_delta(table, i, i - 1);
        let d = d.signum();

        let delta_min = Self::calc_delta(table, i, i - d);
        let mut l_max = 128;

        loop {
            let k = i + (l_max * d);
            if k >= table_size {
                break;
            }

            if !(Self::calc_delta(table, i, k) > delta_min) {
                break;
            }

            l_max *= 4;
        }

        let mut l = 0;

        let mut div = 2;
        loop {
            let t = l_max / div;
            let k = i + ((l + t) * d);
            if Self::calc_delta(table, i, k) > delta_min {
                l += t;
            }

            if t == 1 {
                break;
            }

            div *= 2;
        }

        let j = i + (l * d);
        let delta_node = Self::calc_delta(table, i, j);
        let mut s = 0;

        let mut div = 2;
        loop {
            let t = ceil_div(l, div);
            let k = i + ((s + t) * d);

            if Self::calc_delta(table, i, k) > delta_node {
                s += t;
            }

            if div >= l {
                break;
            }

            div *= 2;
        }

        let gamma = i + (s * d) + d.min(0);

        NodeDivision {
            index_a: i as u32,
            index_b: j as u32,
            split: gamma as u32,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LBVHBuilder {
    scene_aabb: Aabb,
    nodes: Vec<Node>,
}

impl LBVHBuilder {
    pub fn new() -> Self {
        Self {
            scene_aabb: Aabb::new(),
            nodes: Vec::new(),
        }
    }

    fn fit_boxes(nodes: &mut [Node], prims: &[Aabb]) {
        let mut indices: Vec<usize> = Vec::with_capacity(nodes.len());

        indices.push(0);

        let mut i = 0;
        while i < indices.len() {
            let j = indices[i];
            if !nodes[j].left_is_leaf() {
                indices.push(nodes[j].left_leaf_index() as usize);
            }

            if !nodes[j].right_is_leaf() {
                indices.push(nodes[j].right_leaf_index() as usize);
            }

            i += 1;
        }

        i = indices.len();
        while i > 0 {
            let j = indices[i - 1];

            if !nodes[j].left_is_leaf() {
                nodes[j].bounds = nodes[nodes[j].left as usize].bounds;
            } else {
                nodes[j].bounds = prims[nodes[j].left_leaf_index() as usize];
            }

            if !nodes[j].right_is_leaf() {
                nodes[j].bounds = nodes[nodes[j].right as usize].bounds;
            } else {
                nodes[j].bounds = prims[nodes[j].right_leaf_index() as usize];
            }

            i -= 1;
        }
    }
}

impl BVHBuilder for LBVHBuilder {
    fn build(&mut self, aabbs: &[Aabb]) {
        self.scene_aabb = Aabb::new();
        for bb in aabbs {
            self.scene_aabb.grow_bb(bb);
        }

        let mut curve = SpaceFillingCurve::new(aabbs, &self.scene_aabb).sorted();
        self.nodes = vec![Node::new(); curve.len() - 1];

        self.nodes.iter_mut().enumerate().for_each(|(i, node)| {
            let div = NodeDivision::divide_node(&curve, i);
            let l_is_leaf = div.min() == (div.split + 0);
            let r_is_leaf = div.min() == (div.split + 1);

            let l_mask = if l_is_leaf {
                highest_bit()
            } else { 0 };

            let r_mask = if r_is_leaf {
                highest_bit()
            } else { 0 };

            node.left = (div.split + 0) | l_mask;
            node.right = (div.split + 1) | r_mask;
        });

        Self::fit_boxes(self.nodes.as_mut_slice(), aabbs);
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn traverse_t<I: FnMut(usize, f32, f32) -> Option<f32>>(&self, ray: Ray, t_min: f32, t_max: f32, i: I) -> Option<f32> {
        None
    }

    fn occludes<I: FnMut(usize, f32, f32) -> bool>(&self, ray: Ray, t_min: f32, t_max: f32, occludes: I) -> bool {
        false
    }

    fn depth_test<I: Fn(usize, f32, f32) -> Option<(f32, u32)>>(&self, ray: Ray, t_min: f32, t_max: f32, i: I) -> (f32, u32) {
        (0.0, 0)
    }

    fn traverse4<I: FnMut(usize, &mut RayPacket4)>(&self, packet: &mut RayPacket4, mut i: I) {
        let mut hit_stack = [0 as u32; 64];
        let mut stack_ptr: i32 = 0;

        let one = Vec4::one();
        let inv_dir_x = one / Vec4::from(packet.direction_x);
        let inv_dir_y = one / Vec4::from(packet.direction_y);
        let inv_dir_z = one / Vec4::from(packet.direction_z);

        while stack_ptr >= 0 {
            let node = &self.nodes[hit_stack[stack_ptr as usize] as usize];
            stack_ptr -= 1;

            if !node.bounds.intersect4(packet, inv_dir_x, inv_dir_y, inv_dir_z).is_some() {
                continue;
            }

            if node.left_is_leaf() {
                let index = node.left_leaf_index() as usize;
                i(index, packet);
            } else {
                stack_ptr += 1;
                hit_stack[stack_ptr as usize] = node.left;
            }

            if node.right_is_leaf() {
                let index = node.right_leaf_index() as usize;
                i(index, packet);
            } else {
                stack_ptr += 1;
                hit_stack[stack_ptr as usize] = node.right;
            }
        }
    }
}

impl Bounds for LBVHBuilder {
    fn bounds(&self) -> Aabb {
        self.scene_aabb.clone()
    }
}