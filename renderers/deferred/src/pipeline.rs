use super::output::*;
use rfw_scene::{AnimVertexData, VertexData};
use shared::*;
use std::borrow::Cow;

pub struct RenderPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub layout: wgpu::PipelineLayout,
    pub anim_layout: wgpu::PipelineLayout,
    pub anim_pipeline: wgpu::RenderPipeline,
}

impl RenderPipeline {
    pub fn new(
        device: &wgpu::Device,
        uniform_layout: &wgpu::BindGroupLayout,
        instance_layout: &wgpu::BindGroupLayout,
        texture_layout: &wgpu::BindGroupLayout,
        skin_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let vert_shader: &[u8] = include_bytes!("../shaders/mesh.vert.spv");
        let frag_shader: &[u8] = include_bytes!("../shaders/deferred.frag.spv");

        let vert_module = device.create_shader_module(wgpu::ShaderModuleSource::SpirV(Cow::from(
            vert_shader.as_quad_bytes(),
        )));
        let frag_module = device.create_shader_module(wgpu::ShaderModuleSource::SpirV(Cow::from(
            frag_shader.as_quad_bytes(),
        )));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[uniform_layout, instance_layout, texture_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh-pipeline"),
            layout: Some(&pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vert_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &frag_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                clamp_depth: false,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[
                wgpu::ColorStateDescriptor {
                    // Albedo
                    format: DeferredOutput::STORAGE_FORMAT,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
                wgpu::ColorStateDescriptor {
                    // Normal
                    format: DeferredOutput::STORAGE_FORMAT,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
                wgpu::ColorStateDescriptor {
                    // World pos
                    format: DeferredOutput::STORAGE_FORMAT,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
                wgpu::ColorStateDescriptor {
                    // Screen space
                    format: DeferredOutput::STORAGE_FORMAT,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
                wgpu::ColorStateDescriptor {
                    // Mat params
                    format: DeferredOutput::MAT_PARAM_FORMAT,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
            ],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: DeferredOutput::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
            }),
            vertex_state: wgpu::VertexStateDescriptor {
                vertex_buffers: &[
                    wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<VertexData>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[wgpu::VertexAttributeDescriptor {
                            offset: 0,
                            format: wgpu::VertexFormat::Float4,
                            shader_location: 0,
                        }],
                    },
                    wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<VertexData>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[wgpu::VertexAttributeDescriptor {
                            offset: 16,
                            format: wgpu::VertexFormat::Float3,
                            shader_location: 1,
                        }],
                    },
                    wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<VertexData>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[wgpu::VertexAttributeDescriptor {
                            offset: 28,
                            format: wgpu::VertexFormat::Uint,
                            shader_location: 2,
                        }],
                    },
                    wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<VertexData>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[wgpu::VertexAttributeDescriptor {
                            offset: 32,
                            format: wgpu::VertexFormat::Float2,
                            shader_location: 3,
                        }],
                    },
                    wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<VertexData>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[wgpu::VertexAttributeDescriptor {
                            offset: 40,
                            format: wgpu::VertexFormat::Float4,
                            shader_location: 4,
                        }],
                    },
                ],
                index_format: wgpu::IndexFormat::Uint32,
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let vert_shader: &[u8] = include_bytes!("../shaders/mesh_anim.vert.spv");
        let vert_module = device.create_shader_module(wgpu::ShaderModuleSource::SpirV(Cow::from(
            vert_shader.as_quad_bytes(),
        )));

        let anim_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[uniform_layout, instance_layout, texture_layout, skin_layout],
            push_constant_ranges: &[],
        });

        let anim_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("anim-pipeline"),
            layout: Some(&anim_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vert_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &frag_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                clamp_depth: false,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[
                wgpu::ColorStateDescriptor {
                    // Albedo
                    format: DeferredOutput::STORAGE_FORMAT,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
                wgpu::ColorStateDescriptor {
                    // Normal
                    format: DeferredOutput::STORAGE_FORMAT,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
                wgpu::ColorStateDescriptor {
                    // World pos
                    format: DeferredOutput::STORAGE_FORMAT,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
                wgpu::ColorStateDescriptor {
                    // Screen space
                    format: DeferredOutput::STORAGE_FORMAT,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
                wgpu::ColorStateDescriptor {
                    // Mat params
                    format: DeferredOutput::MAT_PARAM_FORMAT,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
            ],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: DeferredOutput::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
            }),
            vertex_state: wgpu::VertexStateDescriptor {
                vertex_buffers: &[
                    wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<VertexData>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[wgpu::VertexAttributeDescriptor {
                            offset: 0,
                            format: wgpu::VertexFormat::Float4,
                            shader_location: 0,
                        }],
                    },
                    wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<VertexData>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[wgpu::VertexAttributeDescriptor {
                            offset: 16,
                            format: wgpu::VertexFormat::Float3,
                            shader_location: 1,
                        }],
                    },
                    wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<VertexData>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[wgpu::VertexAttributeDescriptor {
                            offset: 28,
                            format: wgpu::VertexFormat::Uint,
                            shader_location: 2,
                        }],
                    },
                    wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<VertexData>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[wgpu::VertexAttributeDescriptor {
                            offset: 32,
                            format: wgpu::VertexFormat::Float2,
                            shader_location: 3,
                        }],
                    },
                    wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<VertexData>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[wgpu::VertexAttributeDescriptor {
                            offset: 40,
                            format: wgpu::VertexFormat::Float4,
                            shader_location: 4,
                        }],
                    },
                    wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<AnimVertexData>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[wgpu::VertexAttributeDescriptor {
                            offset: 0,
                            format: wgpu::VertexFormat::Uint4,
                            shader_location: 5,
                        }],
                    },
                    wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<AnimVertexData>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[wgpu::VertexAttributeDescriptor {
                            offset: 16,
                            format: wgpu::VertexFormat::Float4,
                            shader_location: 6,
                        }],
                    },
                ],
                index_format: wgpu::IndexFormat::Uint32,
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Self {
            pipeline,
            layout: pipeline_layout,
            anim_pipeline,
            anim_layout,
        }
    }
}
