use crate::utils::*;
use crate::{Instance, ObjectRef};
use animation::{Animation, Channel};
use glam::*;
use rayon::prelude::*;

#[cfg(feature = "object_caching")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

pub mod animation;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u32)]
pub enum NodeFlags {
    First = 0,
    Transformed = 1,
    Morphed = 2,
}

impl Into<u8> for NodeFlags {
    fn into(self) -> u8 {
        self as u8
    }
}

#[cfg_attr(feature = "object_caching", derive(Serialize, Deserialize))]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct NodeMesh {
    pub object_id: ObjectRef,
    pub instance_id: Option<u32>,
}

impl std::fmt::Display for NodeMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NodeMesh {{ object_id: {}, instance_id: {} }}",
            self.object_id, self.object_id
        )
    }
}

#[derive(Debug, Clone)]
pub struct NodeDescriptor {
    pub name: String,
    pub child_nodes: Vec<NodeDescriptor>,

    pub translation: Vec3A,
    pub rotation: Quat,
    pub scale: Vec3A,

    pub meshes: Vec<ObjectRef>,
    pub skin: Option<SkinDescriptor>,
    pub weights: Vec<f32>,

    /// An ID that is guaranteed to be unique within the scene descriptor this
    /// node descriptor belongs to.
    pub id: u32,
}

#[derive(Debug, Clone)]
pub struct SkinDescriptor {
    pub name: String,
    pub inverse_bind_matrices: Vec<Mat4>,
    // Joint node descriptor IDs (NodeDescriptor::id)
    pub joint_nodes: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct AnimationDescriptor {
    pub name: String,
    // (node descriptor ID, animation channel)
    pub channels: Vec<(u32, Channel)>,
}

#[derive(Debug, Clone)]
pub struct SceneDescriptor {
    pub nodes: Vec<NodeDescriptor>,
    pub animations: Vec<AnimationDescriptor>,
}

#[cfg_attr(feature = "object_caching", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Node {
    translation: Vec3A,
    rotation: Quat,
    scale: Vec3A,
    matrix: Mat4,
    local_matrix: Mat4,
    pub combined_matrix: Mat4,
    pub skin: Option<u32>,
    pub weights: Vec<f32>,
    pub meshes: Vec<NodeMesh>,
    pub child_nodes: Vec<u32>,
    pub name: String,
    pub changed: bool,
    pub first: bool,
    pub morphed: bool,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            translation: Vec3A::zero(),
            rotation: Quat::identity(),
            scale: Vec3A::splat(1.0),
            matrix: Mat4::identity(),
            local_matrix: Mat4::identity(),
            combined_matrix: Mat4::identity(),
            skin: None,
            weights: Vec::new(),
            meshes: Vec::new(),
            child_nodes: Vec::new(),
            changed: true,
            morphed: false,
            first: true,
            name: String::new(),
        }
    }
}

impl Node {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_translation<T: Into<[f32; 3]>>(&mut self, t: T) {
        self.translation = Vec3A::from(t.into());
        self.changed = true;
    }

    /// Set rotation using an xyzw quaternion
    pub fn set_rotation<T: Into<[f32; 4]>>(&mut self, r: T) {
        self.rotation = Quat::from(Vec4::from(r.into()));
        self.changed = true;
    }

    pub fn set_scale<T: Into<[f32; 3]>>(&mut self, s: T) {
        self.scale = Vec3A::from(s.into());
        self.changed = true;
    }

    pub fn set_matrix<T: Into<[f32; 16]>>(&mut self, matrix: T) {
        self.matrix = Mat4::from_cols_array(&matrix.into());
        self.changed = true;
    }

    pub fn set_matrix_cols<T: Into<[[f32; 4]; 4]>>(&mut self, matrix: T) {
        self.matrix = Mat4::from_cols_array_2d(&matrix.into());
        self.changed = true;
    }

    pub fn scale_x(&mut self, scale: f32) {
        self.scale *= Vec3A::new(scale, 1.0, 1.0);
        self.changed = true;
    }

    pub fn scale_y(&mut self, scale: f32) {
        self.scale *= Vec3A::new(1.0, scale, 1.0);
        self.changed = true;
    }

    pub fn scale_z(&mut self, scale: f32) {
        self.scale *= Vec3A::new(1.0, 1.0, scale);
        self.changed = true;
    }

    pub fn scale<T: Into<[f32; 3]>>(&mut self, offset: T) {
        self.scale *= Vec3A::from(offset.into());
        self.changed = true;
    }

    pub fn translate_x(&mut self, offset: f32) {
        self.translation += Vec3A::new(offset, 0.0, 0.0);
        self.changed = true;
    }

    pub fn translate_y(&mut self, offset: f32) {
        self.translation += Vec3A::new(0.0, offset, 0.0);
        self.changed = true;
    }

    pub fn translate_z(&mut self, offset: f32) {
        self.translation += Vec3A::new(0.0, 0.0, offset);
        self.changed = true;
    }

    pub fn translate<T: Into<[f32; 3]>>(&mut self, offset: T) {
        let offset: [f32; 3] = offset.into();
        self.translation += Vec3A::from(offset);
        self.changed = true;
    }

    pub fn rotate<T: Into<[f32; 3]>>(&mut self, degrees: T) {
        let degrees: [f32; 3] = degrees.into();
        self.rotation *= Quat::from_rotation_x(degrees[0].to_radians());
        self.rotation *= Quat::from_rotation_y(degrees[1].to_radians());
        self.rotation *= Quat::from_rotation_z(degrees[2].to_radians());
        self.changed = true;
    }

    pub fn rotate_x(&mut self, degrees: f32) {
        self.rotation *= Quat::from_rotation_x(degrees.to_radians());
        self.changed = true;
    }

    pub fn rotate_y(&mut self, degrees: f32) {
        self.rotation *= Quat::from_rotation_y(degrees.to_radians());
        self.changed = true;
    }

    pub fn rotate_z(&mut self, degrees: f32) {
        self.rotation *= Quat::from_rotation_z(degrees.to_radians());
        self.changed = true;
    }

    pub fn update_matrix(&mut self) {
        let trs = Mat4::from_scale_rotation_translation(
            self.scale.into(),
            self.rotation,
            self.translation.into(),
        );
        self.local_matrix = trs * self.matrix;
        self.changed = false;
    }
}

#[cfg_attr(feature = "object_caching", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct NodeGraph {
    nodes: TrackedStorage<Node>,
    root_nodes: TrackedStorage<u32>,
    pub animations: TrackedStorage<Animation>,
    pub skins: TrackedStorage<Skin>,
    pub active_animation: Option<usize>,
}

impl Default for NodeGraph {
    fn default() -> Self {
        Self {
            nodes: TrackedStorage::new(),
            root_nodes: TrackedStorage::new(),
            animations: TrackedStorage::new(),
            skins: TrackedStorage::new(),
            active_animation: None,
        }
    }
}

impl NodeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn initialize(
        &mut self,
        instances: &RwLock<TrackedStorage<Instance>>,
        skins: &RwLock<TrackedStorage<Skin>>,
    ) {
        for (_, node) in self.nodes.iter_mut() {
            if let Some(skin_id) = node.skin {
                if let Ok(mut skins) = skins.write() {
                    let skin_id = skins.push(self.skins[skin_id as usize].clone());
                    node.skin = Some(skin_id as u32);
                }
            }

            for mesh in &mut node.meshes {
                if mesh.instance_id.is_none() {
                    let mut instances = instances.write().unwrap();
                    let id = instances.allocate();
                    mesh.instance_id = Some(id as u32);
                    instances[id].object_id = mesh.object_id;
                    instances[id].skin_id = node.skin;
                }
            }
        }
    }

    pub fn add_animation(&mut self, anim: Animation) -> usize {
        self.animations.push(anim)
    }

    pub fn add_skin(&mut self, skin: Skin) -> usize {
        self.skins.push(skin)
    }

    pub fn allocate(&mut self) -> usize {
        self.nodes.allocate()
    }

    pub fn any_changed(&self) -> bool {
        self.nodes.any_changed()
    }

    pub fn reset_changed(&mut self) {
        self.nodes.reset_changed();
    }

    pub fn trigger_changed(&mut self, id: usize) {
        self.nodes.trigger_changed(id);
    }

    pub fn trigger_changed_all(&mut self) {
        self.nodes.trigger_changed_all();
        self.root_nodes.trigger_changed_all();
    }

    pub fn add_root_node(&mut self, id: usize) {
        assert!(self.nodes.get(id).is_some());
        self.root_nodes.push(id as u32);
    }

    pub fn update(
        &mut self,
        instances: &RwLock<TrackedStorage<Instance>>,
        skins: &RwLock<TrackedStorage<Skin>>,
    ) -> bool {
        if !self.root_nodes.any_changed() && !self.nodes.any_changed() {
            return false;
        }

        let mut changed = false;
        for i in 0..self.root_nodes.len() {
            let id = self.root_nodes[i] as usize;
            if self.nodes.get_changed(id) {
                self.root_nodes.trigger_changed(i);
            }
        }

        for (_, root_node) in self.root_nodes.iter_changed() {
            changed |= Self::traverse_children(
                (*root_node) as usize,
                Mat4::identity(),
                &mut self.nodes,
                instances,
                skins,
            );
        }

        self.root_nodes.reset_changed();
        self.nodes.reset_changed();

        changed
    }

    pub fn get(&self, index: usize) -> Option<&Node> {
        self.nodes.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut Node> {
        self.nodes.get_mut(index)
    }

    pub unsafe fn get_unchecked(&self, index: usize) -> &Node {
        self.nodes.get_unchecked(index)
    }

    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Node {
        self.nodes.get_unchecked_mut(index)
    }

    pub unsafe fn as_ptr(&self) -> *const Node {
        self.nodes.as_ptr()
    }

    pub unsafe fn as_mut_ptr(&mut self) -> *mut Node {
        self.nodes.as_mut_ptr()
    }

    fn traverse_children(
        current_index: usize,
        accumulated_matrix: Mat4,
        nodes: &mut TrackedStorage<Node>,
        instances: &RwLock<TrackedStorage<Instance>>,
        skins: &RwLock<TrackedStorage<Skin>>,
    ) -> bool {
        let mut changed = nodes[current_index].changed;
        if changed {
            nodes[current_index].update_matrix();
        }

        let combined_matrix = accumulated_matrix * nodes[current_index].local_matrix;
        nodes[current_index].combined_matrix = combined_matrix;

        // Use an unsafe slice to prevent having to copy the vec
        let child_nodes = unsafe {
            std::slice::from_raw_parts(
                nodes[current_index].child_nodes.as_ptr(),
                nodes[current_index].child_nodes.len(),
            )
        };

        // Update children
        for c_id in child_nodes.iter() {
            let c_id = *c_id as usize;
            changed |= Self::traverse_children(c_id, combined_matrix, nodes, instances, skins);
        }

        if !changed && !nodes[current_index].first {
            return false;
        }

        let meshes = &nodes[current_index].meshes;
        meshes
            .iter()
            .filter(|m| m.instance_id.is_some())
            .for_each(|m| {
                if let Ok(mut instances) = instances.write() {
                    instances[m.instance_id.unwrap() as usize].set_transform(combined_matrix);
                }

                // TODO: Morph animations
            });

        // Update skin
        if let Some(skin) = nodes[current_index].skin {
            if let Ok(mut skins) = skins.write() {
                let skin = &mut skins[skin as usize];
                let inverse_transform = combined_matrix.inverse();
                let inverse_bind_matrices = &skin.inverse_bind_matrices;
                let joint_matrices = &mut skin.joint_matrices;

                skin.joint_nodes
                    .iter()
                    .enumerate()
                    .for_each(|(i, node_id)| {
                        let node_id = *node_id as usize;
                        joint_matrices[i] = inverse_transform
                            * nodes[node_id].combined_matrix
                            * inverse_bind_matrices[i];
                    });
            }

            meshes
                .iter()
                .filter(|m| m.instance_id.is_some())
                .for_each(|m| {
                    if let Ok(mut instances) = instances.write() {
                        instances[m.instance_id.unwrap() as usize].skin_id =
                            nodes[current_index].skin;
                    }
                });
        }

        nodes[current_index].changed = false;

        // Return whether this node or its children changed
        changed
    }

    pub fn root_nodes(&self) -> Vec<u32> {
        self.root_nodes.iter().map(|(_, i)| *i).collect()
    }

    pub fn iter_root_nodes(&self) -> RootNodeIterator<'_> {
        RootNodeIterator {
            nodes: unsafe { self.nodes.as_slice() },
            root_nodes: self.root_nodes.iter(),
        }
    }

    pub fn iter_root_nodes_mut(&mut self) -> RootNodeIteratorMut<'_> {
        RootNodeIteratorMut {
            nodes: unsafe { self.nodes.as_mut_slice() },
            root_nodes: self.root_nodes.iter(),
        }
    }

    pub fn update_animation(&mut self, time: f32) {
        if let Some(animation) = self.active_animation {
            self.animations[animation].set_time(time, &mut self.nodes);
        }
    }

    pub fn set_active_animation(&mut self, id: usize) -> Result<(), ()> {
        if let Some(_) = self.animations.get(id) {
            self.active_animation = Some(id);
            return Ok(());
        }

        Err(())
    }

    pub fn load_scene_descriptor(
        &mut self,
        scene_descriptor: &SceneDescriptor,
        instances: &mut TrackedStorage<Instance>,
    ) {
        let mut node_map: HashMap<u32, u32> = HashMap::with_capacity(scene_descriptor.nodes.len());

        let mut root_nodes = Vec::with_capacity(scene_descriptor.nodes.len());
        for node in &scene_descriptor.nodes {
            let node_id =
                self.load_node_descriptor(&mut node_map, node, scene_descriptor, instances);

            root_nodes.push(node_id);
            self.root_nodes.push(node_id);
        }

        for animation in &scene_descriptor.animations {
            let channels = animation
                .channels
                .iter()
                .map(|(node_desc_id, channel)| {
                    let node_id = node_map[&node_desc_id];
                    (node_id, channel.clone())
                })
                .collect();

            let animation_id = self.animations.push(Animation {
                name: animation.name.clone(),
                // TODO
                affected_roots: root_nodes.clone(),
                channels,
                time: 0.0,
            });

            self.set_active_animation(animation_id).unwrap();
            self.update_animation(0.0);
        }
    }

    pub fn load_node_descriptor(
        &mut self,
        node_map: &mut HashMap<u32, u32>,
        descriptor: &NodeDescriptor,
        scene_descriptor: &SceneDescriptor,
        instances: &mut TrackedStorage<Instance>,
    ) -> u32 {
        let child_nodes: Vec<u32> = descriptor
            .child_nodes
            .iter()
            .map(|child_node_descriptor| {
                self.load_node_descriptor(
                    node_map,
                    child_node_descriptor,
                    scene_descriptor,
                    instances,
                )
            })
            .collect();

        let skin_id = descriptor
            .skin
            .as_ref()
            .map(|s| {
                let joint_nodes = s
                    .joint_nodes
                    .iter()
                    .map(|joint_node_id| node_map[joint_node_id])
                    .collect();

                self.skins.push(Skin {
                    name: s.name.clone(),
                    joint_nodes,
                    inverse_bind_matrices: s.inverse_bind_matrices.clone(),
                    joint_matrices: vec![Mat4::identity(); s.inverse_bind_matrices.len()],
                })
            })
            .map(|id| id as u32);

        let meshes: Vec<NodeMesh> = descriptor
            .meshes
            .iter()
            .map(|mesh| {
                let instance_id = instances.allocate();
                instances[instance_id].object_id = *mesh;
                instances[instance_id].skin_id = skin_id;
                NodeMesh {
                    object_id: *mesh,
                    instance_id: Some(instance_id as u32),
                }
            })
            .collect();

        let mut node = Node {
            translation: descriptor.translation,
            rotation: descriptor.rotation,
            scale: descriptor.scale,
            matrix: Mat4::identity(),
            local_matrix: Mat4::identity(),
            combined_matrix: Mat4::identity(),
            skin: skin_id,
            weights: descriptor.weights.clone(),
            meshes,
            child_nodes: child_nodes,
            changed: true,
            morphed: false,
            first: true,
            name: descriptor.name.clone(),
        };
        node.update_matrix();
        let node_id = self.nodes.push(node) as u32;

        node_map.insert(descriptor.id, node_id);

        node_id
    }

    pub fn from_scene_descriptor(
        scene_descriptor: &SceneDescriptor,
        instances: &mut TrackedStorage<Instance>,
    ) -> Self {
        let mut graph = Self::new();
        graph.load_scene_descriptor(scene_descriptor, instances);
        graph
    }

    pub fn from_node_descriptor(
        node_map: &mut HashMap<u32, u32>,
        descriptor: &NodeDescriptor,
        scene_descriptor: &SceneDescriptor,
        instances: &mut TrackedStorage<Instance>,
    ) -> Self {
        let mut graph = Self::new();
        graph.load_node_descriptor(node_map, descriptor, scene_descriptor, instances);
        graph
    }
}

pub struct RootNodeIterator<'a> {
    nodes: &'a [Node],
    root_nodes: FlaggedIterator<'a, u32>,
}

impl<'a> Iterator for RootNodeIterator<'a> {
    type Item = &'a Node;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((_, root_node)) = self.root_nodes.next() {
            Some(&self.nodes[*root_node as usize])
        } else {
            None
        }
    }
}

pub struct RootNodeIteratorMut<'a> {
    nodes: &'a mut [Node],
    root_nodes: FlaggedIterator<'a, u32>,
}

impl<'a> Iterator for RootNodeIteratorMut<'a> {
    type Item = &'a mut Node;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((_, root_node)) = self.root_nodes.next() {
            let ptr = self.nodes.as_mut_ptr();
            let reference = unsafe { ptr.add(*root_node as usize).as_mut().unwrap() };
            Some(reference)
        } else {
            None
        }
    }
}

impl std::ops::Index<usize> for NodeGraph {
    type Output = Node;
    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}

impl std::ops::IndexMut<usize> for NodeGraph {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}

#[cfg_attr(feature = "object_caching", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Skin {
    pub name: String,
    pub joint_nodes: Vec<u32>,
    pub inverse_bind_matrices: Vec<Mat4>,
    pub joint_matrices: Vec<Mat4>,
}

impl Default for Skin {
    fn default() -> Self {
        Self {
            name: String::new(),
            joint_nodes: Vec::new(),
            inverse_bind_matrices: Vec::new(),
            joint_matrices: Vec::new(),
        }
    }
}

#[cfg_attr(feature = "object_caching", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct SceneGraph {
    sub_graphs: TrackedStorage<NodeGraph>,
    times: TrackedStorage<f32>,
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self {
            sub_graphs: TrackedStorage::default(),
            times: TrackedStorage::default(),
        }
    }
}

impl SceneGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn synchronize(
        &mut self,
        instances: &RwLock<TrackedStorage<Instance>>,
        skins: &RwLock<TrackedStorage<Skin>>,
    ) -> bool {
        let times = &self.times;
        self.sub_graphs
            .iter_mut()
            .filter(|(i, _)| times.get_changed(*i))
            .par_bridge()
            .for_each(|(i, g)| {
                let time = times[i];
                g.update_animation(time);
            });

        let changed: u32 = self
            .sub_graphs
            .iter_mut()
            .par_bridge()
            .map(|(_, graph)| {
                if graph.update(instances, skins) {
                    graph.reset_changed();
                    1
                } else {
                    0
                }
            })
            .sum();

        self.times.reset_changed();
        self.sub_graphs.reset_changed();
        changed > 0
    }

    pub fn add_graph(&mut self, graph: NodeGraph) -> u32 {
        let id = self.sub_graphs.push(graph);
        self.times.overwrite(id, 0.0);
        id as u32
    }

    pub fn remove_graph(
        &mut self,
        id: u32,
        instances: &RwLock<TrackedStorage<Instance>>,
        skins: &RwLock<TrackedStorage<Skin>>,
    ) -> bool {
        // Remove instances part of this sub graph
        if let Some(graph) = self.sub_graphs.get(id as usize) {
            for (_, node) in graph.nodes.iter() {
                if let Some(skin_id) = node.skin {
                    skins.write().unwrap().erase(skin_id as usize).unwrap();
                }

                for mesh in &node.meshes {
                    if let Some(id) = mesh.instance_id {
                        // instances.write().unwrap().erase(id as usize).unwrap();
                        let instance = &mut instances.write().unwrap()[id as usize];
                        instance.object_id = ObjectRef::None;
                    }
                }
            }
        }

        match (
            self.sub_graphs.erase(id as usize),
            self.times.erase(id as usize),
        ) {
            (Ok(_), Ok(_)) => true,
            _ => false,
        }
    }

    pub fn set_animation(&mut self, id: u32, time: f32) {
        if let Some(t) = self.times.get_mut(id as usize) {
            *t = time;
        }
    }

    pub fn set_animations(&mut self, time: f32) {
        self.times.iter_mut().for_each(|(_, t)| {
            *t = time;
        });
    }
}
