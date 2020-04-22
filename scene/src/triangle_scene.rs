use crate::objects::*;
use crate::scene::*;
use crate::{loaders, utils::*, Camera, FrustrumG, FrustrumResult, MaterialList};
use bvh::Ray;

use bvh::{Bounds, RayPacket4, ShadowPacket4, AABB, BVH, MBVH};
use glam::*;

use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::{
    collections::HashSet, error::Error, fs::File, io::prelude::*, io::BufReader, path::Path,
};
use loaders::obj;

/// Scene optimized for triangles
/// Does not support objects other than Meshes, but does not require virtual calls because of this.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriangleScene {
    objects: Vec<Mesh>,
    object_references: Vec<HashSet<usize>>,
    instances: Vec<Instance>,
    instance_references: Vec<usize>,
    empty_object_slots: Vec<usize>,
    empty_instance_slots: Vec<usize>,
    bvh: BVH,
    mbvh: MBVH,
    flags: Flags,
    pub materials: MaterialList,
}

pub struct GPUScene {
    pub uniform_buffer: wgpu::Buffer,
    pub staging_buffer: wgpu::Buffer,

    pub render_pipeline: wgpu::RenderPipeline,
    pub render_pipeline_layout: wgpu::PipelineLayout,
    pub uniform_bind_group: wgpu::BindGroup,

    pub instance_bind_groups: Vec<wgpu::BindGroup>,
    pub instance_buffers: Vec<InstanceMatrices>,
    pub vertex_buffers: Vec<VertexBuffer>,

    pub material_buffer: (wgpu::BufferAddress, wgpu::Buffer),
    pub material_textures: Vec<wgpu::Texture>,
    pub material_texture_views: Vec<wgpu::TextureView>,
    pub material_texture_sampler: wgpu::Sampler,
    pub material_bind_groups: Vec<wgpu::BindGroup>,

    pub uniform_bind_group_layout: wgpu::BindGroupLayout,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    pub triangle_bind_group_layout: wgpu::BindGroupLayout,
}

impl GPUScene {
    pub fn new(
        scene: &TriangleScene,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        use wgpu::*;

        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("triangle-uniform-buffer"),
            size: std::mem::size_of::<Mat4>() as BufferAddress,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let staging_buffer = device.create_buffer_mapped(&BufferDescriptor {
            label: Some("staging-buffer"),
            size: std::mem::size_of::<Mat4>() as BufferAddress,
            usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
        });
        staging_buffer.data.copy_from_slice(unsafe {
            let matrix = Mat4::identity();
            std::slice::from_raw_parts(
                matrix.as_ref().as_ptr() as *const u8,
                std::mem::size_of::<Mat4>(),
            )
        });
        let staging_buffer = staging_buffer.finish();

        let uniform_bind_group_layout = Self::create_uniform_bind_group_layout(device);
        let triangle_bind_group_layout = Self::create_instance_bind_group_layout(device);
        let texture_bind_group_layout = Self::create_texture_bind_group_layout(device);
        let (render_pipeline_layout, render_pipeline) = Self::create_render_pipeline(
            device,
            output_format,
            depth_format,
            &uniform_bind_group_layout,
            &triangle_bind_group_layout,
            &texture_bind_group_layout,
        );

        let material_buffer = scene.materials.create_buffer(device, queue);
        let material_texture_sampler = Self::create_texture_sampler(device);

        let uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            bindings: &[
                Binding {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &uniform_buffer,
                        range: 0..64,
                    },
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::Buffer {
                        buffer: &material_buffer.1,
                        range: 0..(material_buffer.0),
                    },
                },
                Binding {
                    binding: 2,
                    resource: BindingResource::Sampler(&material_texture_sampler),
                },
            ],
            label: Some("mesh-bind-group-descriptor"),
        });

        let mut gpu_scene = GPUScene {
            uniform_buffer,
            staging_buffer,
            render_pipeline,
            render_pipeline_layout,
            uniform_bind_group,
            instance_bind_groups: Vec::new(),
            instance_buffers: Vec::new(),
            vertex_buffers: Vec::new(),
            material_buffer,
            material_textures: Vec::new(),
            material_texture_views: Vec::new(),
            material_texture_sampler,
            material_bind_groups: Vec::new(),
            uniform_bind_group_layout,
            texture_bind_group_layout,
            triangle_bind_group_layout,
        };

        gpu_scene.synchronize(scene, device, queue);
        gpu_scene
    }

    pub fn synchronize_meshes(
        &mut self,
        scene: &TriangleScene,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let vertex_buffers = scene.create_vertex_buffers(device, queue);
        let instance_buffers = scene.create_instances_buffer(device, queue);
        let instance_bind_groups = scene.create_bind_groups(
            device,
            &self.triangle_bind_group_layout,
            instance_buffers.as_slice(),
        );

        self.vertex_buffers = vertex_buffers;
        self.instance_buffers = instance_buffers;
        self.instance_bind_groups = instance_bind_groups;
    }

    pub fn synchronize_materials(
        &mut self,
        scene: &TriangleScene,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        use wgpu::*;

        // Materials
        self.material_buffer = scene.materials.create_buffer(device, queue);
        self.material_textures = scene.materials.create_textures(device, queue);
        self.material_texture_views = self
            .material_textures
            .iter()
            .map(|tex| tex.create_default_view())
            .collect();
        self.material_bind_groups = (0..scene.materials.len())
            .map(|i| {
                let material = &scene.materials[i];
                let albedo_tex = material.diffuse_tex.max(0) as usize;
                let normal_tex = material.normal_tex.max(0) as usize;

                let albedo_view = &self.material_texture_views[albedo_tex];
                let normal_view = &self.material_texture_views[normal_tex];

                device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    bindings: &[
                        Binding {
                            binding: 0,
                            resource: BindingResource::TextureView(albedo_view),
                        },
                        Binding {
                            binding: 1,
                            resource: BindingResource::TextureView(normal_view),
                        },
                    ],
                    layout: &self.texture_bind_group_layout,
                })
            })
            .collect();

        self.uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &self.uniform_bind_group_layout,
            bindings: &[
                Binding {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &self.uniform_buffer,
                        range: 0..std::mem::size_of::<Mat4>() as u64,
                    },
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::Buffer {
                        buffer: &self.material_buffer.1,
                        range: 0..(self.material_buffer.0),
                    },
                },
                Binding {
                    binding: 2,
                    resource: BindingResource::Sampler(&self.material_texture_sampler),
                },
            ],
            label: Some("mesh-bind-group-descriptor"),
        });
    }

    pub fn synchronize(
        &mut self,
        scene: &TriangleScene,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.synchronize_meshes(scene, device, queue);
        self.synchronize_materials(scene, device, queue);
    }

    pub fn record_render(
        &self,
        camera: &Camera,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        depth_texture: &wgpu::TextureView,
    ) {
        let mapping = self.staging_buffer.map_write(0, 64);
        let matrix = camera.get_rh_matrix();
        let frustrum: FrustrumG = FrustrumG::from_matrix(matrix);

        device.poll(wgpu::Maintain::Wait);

        if let Ok(mut mapping) = futures::executor::block_on(mapping) {
            let slice = mapping.as_slice();
            slice.copy_from_slice(unsafe {
                std::slice::from_raw_parts(matrix.as_ref().as_ptr() as *const u8, 64)
            });
        }

        encoder.copy_buffer_to_buffer(
            &self.staging_buffer,
            0,
            &self.uniform_buffer,
            0,
            std::mem::size_of::<Mat4>() as u64,
        );

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: output,
                resolve_target: None,
                load_op: wgpu::LoadOp::Clear,
                store_op: wgpu::StoreOp::Store,
                clear_color: wgpu::Color {
                    r: 0.0 as f64,
                    g: 0.0 as f64,
                    b: 0.0 as f64,
                    a: 0.0 as f64,
                },
            }],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                attachment: depth_texture,
                depth_load_op: wgpu::LoadOp::Clear,
                depth_store_op: wgpu::StoreOp::Store,
                clear_depth: 1.0,
                stencil_load_op: wgpu::LoadOp::Clear,
                stencil_store_op: wgpu::StoreOp::Clear,
                clear_stencil: 0,
            }),
        });
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_pipeline(&self.render_pipeline);

        for i in 0..self.instance_buffers.len() {
            let instance_buffers: &InstanceMatrices = &self.instance_buffers[i];
            if instance_buffers.count <= 0 {
                continue;
            }

            let instance_bind_group = &self.instance_bind_groups[i];
            let vb: &VertexBuffer = &self.vertex_buffers[i];

            render_pass.set_bind_group(1, instance_bind_group, &[]);
            render_pass.set_vertex_buffer(0, &vb.buffer, 0, 0);
            render_pass.set_vertex_buffer(1, &vb.buffer, 0, 0);
            render_pass.set_vertex_buffer(2, &vb.buffer, 0, 0);
            render_pass.set_vertex_buffer(3, &vb.buffer, 0, 0);

            for i in 0..instance_buffers.count {
                let bounds = vb.bounds.transformed(instance_buffers.actual_matrices[i]);
                if frustrum.aabb_in_frustrum(&bounds) != FrustrumResult::Outside {
                    let i = i as u32;
                    for mesh in vb.meshes.iter() {
                        if frustrum.aabb_in_frustrum(&mesh.bounds) != FrustrumResult::Outside {
                            render_pass.set_bind_group(
                                2,
                                &self.material_bind_groups[mesh.mat_id as usize],
                                &[],
                            );
                            render_pass.draw(mesh.first..mesh.last, i..(i + 1));
                        }
                    }
                }
            }
        }
    }

    pub fn render(
        &self,
        camera: &Camera,
        device: &wgpu::Device,
        output: &wgpu::TextureView,
        depth_texture: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        let mapping = self.staging_buffer.map_write(0, 64);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-command"),
        });

        let matrix = camera.get_rh_matrix();
        let frustrum: FrustrumG = FrustrumG::from_matrix(matrix);

        device.poll(wgpu::Maintain::Wait);

        if let Ok(mut mapping) = futures::executor::block_on(mapping) {
            let slice = mapping.as_slice();
            slice.copy_from_slice(unsafe {
                std::slice::from_raw_parts(matrix.as_ref().as_ptr() as *const u8, 64)
            });
        }

        encoder.copy_buffer_to_buffer(
            &self.staging_buffer,
            0,
            &self.uniform_buffer,
            0,
            std::mem::size_of::<Mat4>() as u64,
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: output,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color {
                        r: 0.0 as f64,
                        g: 0.0 as f64,
                        b: 0.0 as f64,
                        a: 0.0 as f64,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: depth_texture,
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    clear_depth: 1.0,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Clear,
                    clear_stencil: 0,
                }),
            });
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_pipeline(&self.render_pipeline);

            for i in 0..self.instance_buffers.len() {
                let instance_buffers: &InstanceMatrices = self.instance_buffers.get(i).unwrap();
                if instance_buffers.count <= 0 {
                    continue;
                }

                let instance_bind_group = self.instance_bind_groups.get(i).unwrap();
                let vb: &VertexBuffer = &self.vertex_buffers[i];

                render_pass.set_bind_group(1, instance_bind_group, &[]);
                render_pass.set_vertex_buffer(0, &vb.buffer, 0, 0);
                render_pass.set_vertex_buffer(1, &vb.buffer, 0, 0);
                render_pass.set_vertex_buffer(2, &vb.buffer, 0, 0);
                render_pass.set_vertex_buffer(3, &vb.buffer, 0, 0);

                for i in 0..instance_buffers.count {
                    let bounds = vb.bounds.transformed(instance_buffers.actual_matrices[i]);
                    if frustrum.aabb_in_frustrum(&bounds) != FrustrumResult::Outside {
                        let i = i as u32;
                        for mesh in vb.meshes.iter() {
                            if frustrum.aabb_in_frustrum(&mesh.bounds) != FrustrumResult::Outside {
                                render_pass.set_bind_group(
                                    2,
                                    &self.material_bind_groups[mesh.mat_id as usize],
                                    &[],
                                );
                                render_pass.draw(mesh.first..mesh.last, i..(i + 1));
                            }
                        }
                    }
                }
            }
        }

        encoder.finish()
    }

    fn create_instance_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use wgpu::*;
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                BindGroupLayoutEntry {
                    // Instance matrices
                    binding: 0,
                    visibility: ShaderStage::VERTEX,
                    ty: BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: true,
                    },
                },
                BindGroupLayoutEntry {
                    // Instance inverse matrices
                    binding: 1,
                    visibility: ShaderStage::VERTEX,
                    ty: BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: true,
                    },
                },
            ],
            label: Some("mesh-bind-group-descriptor-layout"),
        })
    }

    fn create_uniform_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use wgpu::*;
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            bindings: &[
                BindGroupLayoutEntry {
                    // Matrix buffer
                    binding: 0,
                    visibility: ShaderStage::VERTEX,
                    ty: BindingType::UniformBuffer { dynamic: false },
                },
                BindGroupLayoutEntry {
                    // Material buffer
                    binding: 1,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::StorageBuffer {
                        readonly: true,
                        dynamic: false,
                    },
                },
                BindGroupLayoutEntry {
                    // Texture sampler
                    binding: 2,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Sampler { comparison: false },
                },
            ],
            label: Some("uniform-layout"),
        })
    }

    fn create_texture_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use wgpu::*;
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("texture-bind-group-layout"),
            bindings: &[
                BindGroupLayoutEntry {
                    // Albedo texture
                    binding: 0,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::SampledTexture {
                        component_type: TextureComponentType::Uint,
                        multisampled: false,
                        dimension: TextureViewDimension::D2,
                    },
                },
                BindGroupLayoutEntry {
                    // Normal texture
                    binding: 1,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::SampledTexture {
                        component_type: TextureComponentType::Uint,
                        multisampled: false,
                        dimension: TextureViewDimension::D2,
                    },
                },
            ],
        })
    }

    fn create_texture_sampler(device: &wgpu::Device) -> wgpu::Sampler {
        use wgpu::*;
        device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 4.0,
            compare: CompareFunction::Undefined,
        })
    }

    fn create_render_pipeline(
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        uniform_layout: &wgpu::BindGroupLayout,
        triangle_layout: &wgpu::BindGroupLayout,
        texture_layout: &wgpu::BindGroupLayout,
    ) -> (wgpu::PipelineLayout, wgpu::RenderPipeline) {
        use shaderc::*;
        use wgpu::*;

        let mut compiler = shaderc::Compiler::new().unwrap();

        let vert_shader = include_str!("../../shaders/mesh.vert");
        let frag_shader = include_str!("../../shaders/mesh.frag");

        let mut compile_options = shaderc::CompileOptions::new().unwrap();
        if cfg!(debug_assertions) {
            compile_options.set_optimization_level(OptimizationLevel::Zero);
            compile_options.set_generate_debug_info();
        } else {
            compile_options.set_optimization_level(OptimizationLevel::Performance);
        }

        let vert_shader = match compiler.compile_into_spirv(
            vert_shader,
            ShaderKind::Vertex,
            "shaders/mesh.vert",
            "main",
            Some(&compile_options),
        ) {
            Ok(shader) => shader,
            Err(e) => {
                panic!("Could not compile shaders/mesh.vert: {}", e);
            }
        };
        let frag_shader = match compiler.compile_into_spirv(
            frag_shader,
            ShaderKind::Fragment,
            "shaders/mesh.frag",
            "main",
            Some(&compile_options),
        ) {
            Ok(shader) => shader,
            Err(e) => {
                panic!("Could not compile shaders/mesh.frag: {}", e);
            }
        };

        let vert_module = device.create_shader_module(vert_shader.as_binary());
        let frag_module = device.create_shader_module(frag_shader.as_binary());

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: &[&uniform_layout, &triangle_layout, &texture_layout],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: ProgrammableStageDescriptor {
                module: &vert_module,
                entry_point: "main",
            },
            fragment_stage: Some(ProgrammableStageDescriptor {
                module: &frag_module,
                entry_point: "main",
            }),
            rasterization_state: Some(RasterizationStateDescriptor {
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: PrimitiveTopology::TriangleList,
            color_states: &[ColorStateDescriptor {
                format: output_format,
                alpha_blend: BlendDescriptor::REPLACE,
                color_blend: BlendDescriptor::REPLACE,
                write_mask: ColorWrite::ALL,
            }],
            depth_stencil_state: Some(DepthStencilStateDescriptor {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil_front: StencilStateFaceDescriptor::IGNORE,
                stencil_back: StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            vertex_state: VertexStateDescriptor {
                vertex_buffers: &[
                    VertexBufferDescriptor {
                        stride: std::mem::size_of::<VertexData>() as BufferAddress,
                        step_mode: InputStepMode::Vertex,
                        attributes: &[VertexAttributeDescriptor {
                            offset: 0,
                            format: VertexFormat::Float4,
                            shader_location: 0,
                        }],
                    },
                    VertexBufferDescriptor {
                        stride: std::mem::size_of::<VertexData>() as BufferAddress,
                        step_mode: InputStepMode::Vertex,
                        attributes: &[VertexAttributeDescriptor {
                            offset: 16,
                            format: VertexFormat::Float3,
                            shader_location: 1,
                        }],
                    },
                    VertexBufferDescriptor {
                        stride: std::mem::size_of::<VertexData>() as BufferAddress,
                        step_mode: InputStepMode::Vertex,
                        attributes: &[VertexAttributeDescriptor {
                            offset: 28,
                            format: VertexFormat::Uint,
                            shader_location: 2,
                        }],
                    },
                    VertexBufferDescriptor {
                        stride: std::mem::size_of::<VertexData>() as BufferAddress,
                        step_mode: InputStepMode::Vertex,
                        attributes: &[VertexAttributeDescriptor {
                            offset: 32,
                            format: VertexFormat::Float2,
                            shader_location: 3,
                        }],
                    },
                ],
                index_format: IndexFormat::Uint32,
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        (pipeline_layout, pipeline)
    }
}

pub struct InstanceMatrices {
    pub count: usize,
    pub actual_matrices: Vec<Mat4>,
    pub matrices: wgpu::Buffer,
    pub inverse_matrices: wgpu::Buffer,
}

#[allow(dead_code)]
impl TriangleScene {
    const FF_EXTENSION: &'static str = ".scenev1";

    pub fn new() -> TriangleScene {
        TriangleScene {
            objects: Vec::new(),
            object_references: Vec::new(),
            instances: Vec::new(),
            instance_references: Vec::new(),
            empty_object_slots: Vec::new(),
            empty_instance_slots: Vec::new(),
            bvh: BVH::empty(),
            mbvh: MBVH::empty(),
            flags: Flags::new(),
            materials: MaterialList::new(),
        }
    }

    pub fn load_mesh<S: AsRef<Path>>(&mut self, path: S) -> Option<usize> {
        let path = path.as_ref();
        let extension = path.extension();
        if extension.is_none() {
            return None;
        }

        let extension = extension.unwrap();
        let materials = &mut self.materials;

        if extension == "obj" {
            let obj = obj::Obj::new(path, materials);
            if obj.is_err() {
                println!("Obj error: {}", obj.err().unwrap());
                return None;
            }

            let obj = obj.unwrap();
            let mesh = obj.into_mesh();
            let result = self.add_object(mesh);
            return Some(result);
        }

        None
    }

    pub fn create_instances_buffer(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Vec<InstanceMatrices> {
        use wgpu::*;
        (0..self.objects.len())
            .map(|i| {
                let refs = &self.object_references[i];
                let label = format!("object-{}-instances", i);
                if refs.is_empty() {
                    let matrix = Mat4::identity();

                    let size = std::mem::size_of::<Mat4>();
                    let buffer = device.create_buffer(&BufferDescriptor {
                        label: Some(label.as_str()),
                        size: size as BufferAddress,
                        usage: BufferUsage::STORAGE_READ,
                    });

                    let staging_buffer = device.create_buffer_with_data(
                        unsafe {
                            std::slice::from_raw_parts(matrix.as_ref().as_ptr() as *const u8, size)
                        },
                        BufferUsage::COPY_SRC,
                    );

                    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                        label: Some(label.as_str()),
                    });
                    encoder.copy_buffer_to_buffer(
                        &staging_buffer,
                        0,
                        &buffer,
                        0,
                        size as BufferAddress,
                    );
                    queue.submit(&[encoder.finish()]);

                    let inverse_buffer = device.create_buffer(&BufferDescriptor {
                        label: Some(label.as_str()),
                        size: size as BufferAddress,
                        usage: BufferUsage::STORAGE_READ,
                    });

                    let staging_buffer = device.create_buffer_with_data(
                        unsafe {
                            std::slice::from_raw_parts(matrix.as_ref().as_ptr() as *const u8, size)
                        },
                        BufferUsage::COPY_SRC,
                    );

                    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                        label: Some(label.as_str()),
                    });

                    encoder.copy_buffer_to_buffer(
                        &staging_buffer,
                        0,
                        &buffer,
                        0,
                        size as BufferAddress,
                    );
                    queue.submit(&[encoder.finish()]);

                    InstanceMatrices {
                        count: 0,
                        actual_matrices: Vec::new(),
                        matrices: buffer,
                        inverse_matrices: inverse_buffer,
                    }
                } else {
                    let mut instances: Vec<Mat4> = Vec::with_capacity(refs.len());
                    let mut inverse_instances: Vec<Mat4> = Vec::with_capacity(refs.len());
                    for r in refs {
                        instances.push(self.instances[*r].get_transform());
                        inverse_instances.push(self.instances[*r].get_normal_transform());
                    }

                    let size = instances.len() * std::mem::size_of::<Mat4>();
                    let buffer = device.create_buffer(&BufferDescriptor {
                        label: Some(label.as_str()),
                        size: size as BufferAddress,
                        usage: BufferUsage::STORAGE_READ | BufferUsage::COPY_DST,
                    });

                    let staging_buffer = device.create_buffer_with_data(
                        unsafe {
                            std::slice::from_raw_parts(instances.as_ptr() as *const u8, size)
                        },
                        BufferUsage::COPY_SRC,
                    );

                    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                        label: Some(label.as_str()),
                    });
                    encoder.copy_buffer_to_buffer(
                        &staging_buffer,
                        0,
                        &buffer,
                        0,
                        size as BufferAddress,
                    );
                    queue.submit(&[encoder.finish()]);

                    let inverse_buffer = device.create_buffer(&BufferDescriptor {
                        label: Some(label.as_str()),
                        size: size as BufferAddress,
                        usage: BufferUsage::STORAGE_READ | BufferUsage::COPY_DST,
                    });

                    let staging_buffer = device.create_buffer_with_data(
                        unsafe {
                            std::slice::from_raw_parts(
                                inverse_instances.as_ptr() as *const u8,
                                size,
                            )
                        },
                        BufferUsage::COPY_SRC,
                    );

                    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                        label: Some(label.as_str()),
                    });

                    encoder.copy_buffer_to_buffer(
                        &staging_buffer,
                        0,
                        &inverse_buffer,
                        0,
                        size as BufferAddress,
                    );
                    queue.submit(&[encoder.finish()]);

                    InstanceMatrices {
                        count: instances.len(),
                        actual_matrices: instances,
                        matrices: buffer,
                        inverse_matrices: inverse_buffer,
                    }
                }
            })
            .collect()
    }

    pub fn create_bind_group_layout(&self, device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use wgpu::*;
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                BindGroupLayoutEntry {
                    // Instance matrices
                    binding: 0,
                    visibility: ShaderStage::VERTEX,
                    ty: BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: true,
                    },
                },
                BindGroupLayoutEntry {
                    // Instance inverse matrices
                    binding: 1,
                    visibility: ShaderStage::VERTEX,
                    ty: BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: true,
                    },
                },
            ],
            label: Some("mesh-bind-group-descriptor-layout"),
        })
    }

    pub fn create_bind_groups(
        &self,
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        buffers: &[InstanceMatrices],
    ) -> Vec<wgpu::BindGroup> {
        use wgpu::*;
        buffers
            .into_iter()
            .enumerate()
            .map(|(i, buffers)| {
                device.create_bind_group(&BindGroupDescriptor {
                    layout: bind_group_layout,
                    bindings: &[
                        Binding {
                            binding: 0,
                            resource: BindingResource::Buffer {
                                buffer: &buffers.matrices,
                                range: 0..(buffers.count * std::mem::size_of::<Mat4>())
                                    as BufferAddress,
                            },
                        },
                        Binding {
                            binding: 1,
                            resource: BindingResource::Buffer {
                                buffer: &buffers.inverse_matrices,
                                range: 0..(buffers.count * std::mem::size_of::<Mat4>())
                                    as BufferAddress,
                            },
                        },
                    ],
                    label: Some(format!("mesh-bind-group-{}", i).as_str()),
                })
            })
            .collect()
    }

    pub fn create_vertex_buffers(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Vec<VertexBuffer> {
        use wgpu::*;
        let mut buffers = Vec::with_capacity(self.objects.len());

        for (i, object) in self.objects.iter().enumerate() {
            let object: &Mesh = object;
            let size = object.buffer_size();
            let label = format!("object-{}", i);

            let triangle_buffer = device.create_buffer(&BufferDescriptor {
                label: Some(label.as_str()),
                size: size as BufferAddress,
                usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            });

            let staging_buffer =
                device.create_buffer_with_data(object.as_bytes(), BufferUsage::COPY_SRC);

            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some(label.as_str()),
            });
            encoder.copy_buffer_to_buffer(
                &staging_buffer,
                0,
                &triangle_buffer,
                0,
                size as BufferAddress,
            );
            queue.submit(&[encoder.finish()]);

            buffers.push(VertexBuffer {
                count: object.len(),
                size_in_bytes: object.buffer_size(),
                buffer: triangle_buffer,
                bounds: object.bounds(),
                meshes: object.meshes.clone(),
            });
        }

        buffers
    }

    pub fn get_objects(&self) -> &[Mesh] {
        self.objects.as_slice()
    }

    pub fn get_object<T>(&self, index: usize, mut cb: T)
    where
        T: FnMut(Option<&Mesh>),
    {
        cb(self.objects.get(index));
    }

    pub fn get_object_mut<T>(&mut self, index: usize, mut cb: T)
    where
        T: FnMut(Option<&mut Mesh>),
    {
        cb(self.objects.get_mut(index));
        self.flags.set_flag(SceneFlags::Dirty);
    }

    pub fn add_object(&mut self, object: Mesh) -> usize {
        if !self.empty_object_slots.is_empty() {
            let new_index = self.empty_object_slots.pop().unwrap();
            self.objects[new_index] = object;
            self.object_references[new_index] = HashSet::new();
            return new_index;
        }

        self.objects.push(object);
        self.object_references.push(HashSet::new());
        self.flags.set_flag(SceneFlags::Dirty);
        self.objects.len() - 1
    }

    pub fn set_object(&mut self, index: usize, object: Mesh) -> Result<(), ()> {
        if self.objects.get(index).is_none() {
            return Err(());
        }

        self.objects[index] = object;
        let object_refs = self.object_references[index].clone();
        for i in object_refs {
            self.remove_instance(i).unwrap();
        }

        self.object_references[index].clear();
        self.flags.set_flag(SceneFlags::Dirty);
        Ok(())
    }

    pub fn remove_object(&mut self, object: usize) -> Result<(), ()> {
        if self.objects.get(object).is_none() {
            return Err(());
        }

        self.objects[object] = Mesh::empty();
        let object_refs = self.object_references[object].clone();
        for i in object_refs {
            self.remove_instance(i).unwrap();
        }

        self.object_references[object].clear();
        self.empty_object_slots.push(object);
        self.flags.set_flag(SceneFlags::Dirty);
        Ok(())
    }

    pub fn add_instance(&mut self, index: usize, transform: Mat4) -> Result<usize, ()> {
        let instance_index = {
            if self.objects.get(index).is_none() || self.object_references.get(index).is_none() {
                return Err(());
            }

            if !self.empty_instance_slots.is_empty() {
                let new_index = self.empty_instance_slots.pop().unwrap();
                self.instances[new_index] =
                    Instance::new(index as isize, &self.objects[index].bounds(), transform);
                self.instance_references[new_index] = index;
                return Ok(new_index);
            }

            self.instances.push(Instance::new(
                index as isize,
                &self.objects[index].bounds(),
                transform,
            ));
            self.instances.len() - 1
        };
        self.instance_references.push(index);

        self.object_references[index].insert(instance_index);
        self.flags.set_flag(SceneFlags::Dirty);
        self.flags.set_flag(SceneFlags::Dirty);
        Ok(instance_index)
    }

    pub fn set_instance_object(&mut self, instance: usize, obj_index: usize) -> Result<(), ()> {
        if self.objects.get(obj_index).is_none() || self.instances.get(instance).is_none() {
            return Err(());
        }

        let old_obj_index = self.instance_references[instance];
        self.object_references[old_obj_index].remove(&instance);
        self.instances[instance] = Instance::new(
            obj_index as isize,
            &self.objects[obj_index].bounds(),
            self.instances[instance].get_transform(),
        );
        self.object_references[obj_index].insert(instance);
        self.instance_references[instance] = obj_index;
        self.flags.set_flag(SceneFlags::Dirty);
        Ok(())
    }

    pub fn remove_instance(&mut self, index: usize) -> Result<(), ()> {
        if self.instances.get(index).is_none() {
            return Err(());
        }

        let old_obj_index = self.instance_references[index];
        if self.object_references.get(old_obj_index).is_some() {
            self.object_references[old_obj_index].remove(&index);
        }

        self.instances[index] = Instance::new(
            -1,
            &self.objects[index].bounds(),
            self.instances[index].get_transform(),
        );
        self.instance_references[index] = std::usize::MAX;
        self.empty_instance_slots.push(index);
        self.flags.set_flag(SceneFlags::Dirty);
        Ok(())
    }

    pub fn build_bvh(&mut self) {
        if self.flags.has_flag(SceneFlags::Dirty) {
            // Need to rebuild bvh
            let aabbs: Vec<AABB> = self
                .instances
                .iter()
                .map(|o| o.bounds())
                .collect::<Vec<AABB>>();
            self.bvh = BVH::construct(aabbs.as_slice());
            self.mbvh = MBVH::construct(&self.bvh);
        }
    }

    pub fn serialize<S: AsRef<Path>>(&self, path: S) -> Result<(), Box<dyn Error>> {
        let encoded: Vec<u8> = bincode::serialize(self)?;

        let mut output = OsString::from(path.as_ref().as_os_str());
        output.push(Self::FF_EXTENSION);

        let mut file = File::create(output)?;
        file.write_all(encoded.as_ref())?;
        Ok(())
    }

    pub fn deserialize<S: AsRef<Path>>(path: S) -> Result<Self, Box<dyn Error>> {
        let mut input = OsString::from(path.as_ref().as_os_str());
        input.push(Self::FF_EXTENSION);
        let file = File::open(input)?;
        let reader = BufReader::new(file);
        let object: Self = bincode::deserialize_from(reader)?;
        Ok(object)
    }

    pub fn create_gpu_scene(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> GPUScene {
        GPUScene::new(&self, device, queue, output_format, depth_format)
    }

    pub fn create_intersector(&self) -> TriangleIntersector {
        TriangleIntersector {
            objects: self.objects.as_slice(),
            instances: self.instances.as_slice(),
            bvh: &self.bvh,
            mbvh: &self.mbvh,
        }
    }
}

#[derive(Copy, Clone)]
pub struct TriangleIntersector<'a> {
    objects: &'a [Mesh],
    instances: &'a [Instance],
    bvh: &'a BVH,
    mbvh: &'a MBVH,
}

impl<'a> TriangleIntersector<'a> {
    pub fn occludes(&self, ray: Ray, t_min: f32, t_max: f32) -> bool {
        let (origin, direction) = ray.into();

        let intersection = |i, t_min, t_max| {
            let instance = &self.instances[i as usize];
            if let Some((origin, direction)) = instance.intersects(ray, t_max) {
                return self.objects[instance.get_hit_id() as usize].occludes(
                    (origin, direction).into(),
                    t_min,
                    t_max,
                );
            }
            false
        };

        let bvh = self.bvh;
        let mbvh = self.mbvh;

        unsafe {
            return match USE_MBVH {
                true => mbvh.occludes(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection,
                ),
                _ => bvh.occludes(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection,
                ),
            };
        }
    }

    pub fn intersect(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let (origin, direction) = ray.into();

        let mut instance_id = -1;
        let intersection = |i, t_min, t_max| {
            let instance = &self.instances[i as usize];
            if let Some((origin, direction)) = instance.intersects(ray, t_max) {
                if let Some(hit) = self.objects[instance.get_hit_id() as usize].intersect(
                    (origin, direction).into(),
                    t_min,
                    t_max,
                ) {
                    instance_id = i as i32;
                    return Some((hit.t, hit));
                }
            }
            None
        };

        let hit = unsafe {
            match USE_MBVH {
                true => self.mbvh.traverse(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection,
                ),
                _ => self.bvh.traverse(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection,
                ),
            }
        };

        hit.and_then(|hit| Some(self.instances[instance_id as usize].transform_hit(hit)))
    }

    pub fn intersect_t(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<f32> {
        let (origin, direction) = ray.into();

        let intersection = |i, t_min, t_max| {
            let instance = &self.instances[i as usize];
            if let Some((origin, direction)) = instance.intersects(ray, t_max) {
                return self.objects[instance.get_hit_id() as usize].intersect_t(
                    (origin, direction).into(),
                    t_min,
                    t_max,
                );
            }
            None
        };

        unsafe {
            return match USE_MBVH {
                true => self.mbvh.traverse_t(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection,
                ),
                _ => self.bvh.traverse_t(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection,
                ),
            };
        }
    }

    pub fn depth_test(&self, ray: Ray, t_min: f32, t_max: f32) -> (f32, u32) {
        let (origin, direction) = ray.into();

        let intersection = |i, t_min, t_max| -> Option<(f32, u32)> {
            let instance = &self.instances[i as usize];
            if let Some((origin, direction)) = instance.intersects(ray, t_max) {
                return self.objects[instance.get_hit_id() as usize].depth_test(
                    (origin, direction).into(),
                    t_min,
                    t_max,
                );
            }
            None
        };

        unsafe {
            return match USE_MBVH {
                true => self.mbvh.depth_test(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection,
                ),
                _ => self.bvh.depth_test(
                    origin.as_ref(),
                    direction.as_ref(),
                    t_min,
                    t_max,
                    intersection,
                ),
            };
        }
    }

    pub fn occludes4(&self, _packet: ShadowPacket4) -> [bool; 4] {
        [true; 4]
    }

    pub fn intersect4(
        &self,
        packet: &mut RayPacket4,
        t_min: [f32; 4],
    ) -> ([InstanceID; 4], [PrimID; 4]) {
        let mut instance_ids = [-1 as InstanceID; 4];
        let mut prim_ids = [-1 as PrimID; 4];

        let intersection = |instance_id, packet: &mut RayPacket4| {
            let instance_id = instance_id as usize;
            let instance = &self.instances[instance_id];
            if let Some(mut new_packet) = instance.intersects4(packet) {
                let object = &self.objects[instance.get_hit_id()];
                if let Some(hit) = object.intersect4(&mut new_packet, &t_min) {
                    for i in 0..4 {
                        if hit[i] >= 0 {
                            instance_ids[i] = instance_id as i32;
                            prim_ids[i] = hit[i];
                            packet.t[i] = new_packet.t[i];
                        }
                    }
                }
            }
        };

        unsafe {
            match USE_MBVH {
                true => self.mbvh.traverse4(packet, intersection),
                _ => self.bvh.traverse4(packet, intersection),
            }
        };

        (instance_ids, prim_ids)
    }

    pub fn get_hit_record(
        &self,
        ray: Ray,
        t: f32,
        instance_id: InstanceID,
        prim_id: PrimID,
    ) -> HitRecord {
        let instance: &Instance = &self.instances[instance_id as usize];
        let object_id: usize = instance.get_hit_id();
        let ray = instance.transform_ray(ray);
        instance.transform_hit(self.objects[object_id].get_hit_record(ray, t, prim_id as u32))
    }
}
