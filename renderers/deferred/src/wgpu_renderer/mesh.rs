use super::CopyCommand;
use rtbvh::AABB;
use scene::{AnimVertexData, AnimatedMesh, Mesh, VertexData, VertexMesh};
use shared::BytesConversion;

pub struct VertexBuffer {
    pub count: usize,
    pub size_in_bytes: usize,
    pub buffer: wgpu::Buffer,
    pub bounds: AABB,
    pub meshes: Vec<VertexMesh>,
}

#[derive(Debug)]
pub struct DeferredAnimMesh {
    pub sub_meshes: Vec<VertexMesh>,
    pub vertex_data: Vec<VertexData>,
    pub anim_vertex_data: Vec<AnimVertexData>,
    pub buffer: Option<wgpu::Buffer>,
    pub buffer_size: wgpu::BufferAddress,
    pub anim_buffer: Option<wgpu::Buffer>,
    pub anim_buffer_size: wgpu::BufferAddress,
}

impl Clone for DeferredAnimMesh {
    fn clone(&self) -> Self {
        Self {
            sub_meshes: self.sub_meshes.clone(),
            vertex_data: self.vertex_data.clone(),
            anim_vertex_data: self.anim_vertex_data.clone(),
            buffer: None,
            buffer_size: 0,
            anim_buffer: None,
            anim_buffer_size: 0,
        }
    }
}

impl Default for DeferredAnimMesh {
    fn default() -> Self {
        Self {
            sub_meshes: Vec::new(),
            vertex_data: Vec::new(),
            anim_vertex_data: Vec::new(),
            buffer: None,
            buffer_size: 0,
            anim_buffer: None,
            anim_buffer_size: 0,
        }
    }
}

#[derive(Debug)]
pub struct DeferredMesh {
    pub sub_meshes: Vec<VertexMesh>,
    pub vertex_data: Vec<VertexData>,
    pub buffer: Option<wgpu::Buffer>,
    pub buffer_size: wgpu::BufferAddress,
}

impl Default for DeferredMesh {
    fn default() -> Self {
        Self {
            sub_meshes: Vec::new(),
            vertex_data: Vec::new(),
            buffer: None,
            buffer_size: 0,
        }
    }
}

impl Clone for DeferredMesh {
    fn clone(&self) -> Self {
        Self {
            sub_meshes: self.sub_meshes.clone(),
            vertex_data: self.vertex_data.clone(),
            buffer: None,
            buffer_size: 0,
        }
    }
}

impl DeferredMesh {
    pub fn new(device: &wgpu::Device, mesh: &Mesh) -> Self {
        let buffer_size = mesh.buffer_size() as wgpu::BufferAddress;
        assert!(buffer_size > 0);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(mesh.name.as_str()),
            size: buffer_size,
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        });

        let sub_meshes = mesh.meshes.clone();
        let vertex_data = mesh.vertices.clone();

        Self {
            sub_meshes,
            vertex_data,
            buffer: Some(buffer),
            buffer_size,
        }
    }

    pub fn len(&self) -> usize {
        self.vertex_data.len()
    }

    pub fn get_copy_command(&self, device: &wgpu::Device) -> CopyCommand {
        let data = unsafe {
            std::slice::from_raw_parts(
                self.vertex_data.as_ptr() as *const u8,
                self.vertex_data.len() * std::mem::size_of::<VertexData>(),
            )
        };

        let staging_buffer = device.create_buffer_with_data(data, wgpu::BufferUsage::COPY_SRC);

        CopyCommand {
            destination_buffer: self.buffer.as_ref().unwrap(),
            copy_size: self.buffer_size as wgpu::BufferAddress,
            staging_buffer,
        }
    }
}

impl DeferredAnimMesh {
    pub fn new(device: &wgpu::Device, mesh: &AnimatedMesh) -> Self {
        let buffer_size = mesh.vertices.to_bytes().len() as wgpu::BufferAddress;
        assert!(buffer_size > 0);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(mesh.name.as_str()),
            size: buffer_size,
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        });

        let anim_buffer_size = mesh.anim_vertex_data.to_bytes().len() as wgpu::BufferAddress;
        assert!(anim_buffer_size > 0);
        let anim_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(mesh.name.as_str()),
            size: anim_buffer_size,
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        });

        let sub_meshes = mesh.meshes.clone();
        let vertex_data = mesh.vertices.clone();
        let anim_vertex_data = mesh.anim_vertex_data.clone();

        Self {
            sub_meshes,
            vertex_data,
            anim_vertex_data,
            buffer: Some(buffer),
            buffer_size,
            anim_buffer: Some(anim_buffer),
            anim_buffer_size,
        }
    }

    pub fn len(&self) -> usize {
        self.vertex_data.len()
    }

    pub fn get_copy_command(&self, device: &wgpu::Device) -> (CopyCommand, CopyCommand) {
        let staging_buffer = device.create_buffer_with_data(self.vertex_data.to_bytes(), wgpu::BufferUsage::COPY_SRC);

        let command1 = CopyCommand {
            destination_buffer: self.buffer.as_ref().unwrap(),
            copy_size: self.buffer_size as wgpu::BufferAddress,
            staging_buffer,
        };

        let staging_buffer = device.create_buffer_with_data(self.anim_vertex_data.to_bytes(), wgpu::BufferUsage::COPY_SRC);

        let command2 = CopyCommand {
            destination_buffer: self.anim_buffer.as_ref().unwrap(),
            copy_size: self.anim_buffer_size as wgpu::BufferAddress,
            staging_buffer,
        };

        (command1, command2)
    }
}
