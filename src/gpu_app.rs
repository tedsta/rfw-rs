use fb_template::{shader::*, DeviceFramebuffer, KeyCode, KeyHandler, MouseButtonHandler, Request};
use glam::*;

use crate::utils::*;
use scene::{Camera, GPUScene, TriangleScene};
use std::collections::VecDeque;

pub struct GPUApp<'a> {
    width: u32,
    height: u32,
    compiler: Compiler<'a>,
    blit_pipeline: Option<wgpu::RenderPipeline>,
    blit_bind_group_layout: Option<wgpu::BindGroupLayout>,
    blit_bind_group: Option<wgpu::BindGroup>,

    output_texture: Option<wgpu::Texture>,
    output_texture_view: Option<wgpu::TextureView>,
    output_sampler: Option<wgpu::Sampler>,

    depth_texture: Option<wgpu::Texture>,
    depth_texture_view: Option<wgpu::TextureView>,

    gpu_scene: Option<GPUScene>,
    scene: TriangleScene,
    camera: Camera,
    timer: Timer,
    sc_format: wgpu::TextureFormat,
    fps: Averager<f32>,
    frame_time: Averager<f32>,
}

impl<'a> GPUApp<'a> {
    pub const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new() -> Self {
        let compiler = CompilerBuilder::new()
            .with_opt_level(OptimizationLevel::Zero)
            .with_warnings_as_errors()
            .with_include_dir("shaders")
            .build();

        let scene = TriangleScene::new();

        Self {
            width: 1,
            height: 1,
            compiler,
            blit_pipeline: None,
            blit_bind_group_layout: None,
            blit_bind_group: None,
            output_texture: None,
            output_texture_view: None,
            output_sampler: None,
            depth_texture: None,
            depth_texture_view: None,
            scene,
            gpu_scene: None,
            camera: Camera::zero(),
            timer: Timer::new(),
            sc_format: Self::OUTPUT_FORMAT,
            fps: Averager::with_capacity(25),
            frame_time: Averager::with_capacity(25),
        }
    }
}

impl<'a> DeviceFramebuffer for GPUApp<'a> {
    fn init(
        &mut self,
        width: u32,
        height: u32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sc_format: wgpu::TextureFormat,
        _requests: &mut VecDeque<Request>,
    ) {
        use wgpu::*;

        self.width = width;
        self.height = height;
        self.camera.resize(width, height);
        self.sc_format = sc_format;

        if let Ok(scene) = TriangleScene::deserialize("models/sponza.scene") {
            println!("Loaded scene from cached file: models/sponza.scene");
            self.scene = scene;
        } else {
            let object = self
                .scene
                .load_mesh("models/sponza/sponza.obj")
                .expect("Could not load sponza.obj");

            let _object = self.scene.add_instance(object, Mat4::identity()).unwrap();
            self.scene.serialize("models/sponza.scene").unwrap();
        }

        let gpu_scene = GPUScene::new(
            &self.scene,
            device,
            queue,
            Self::OUTPUT_FORMAT,
            Self::DEPTH_FORMAT,
        );
        self.gpu_scene = Some(gpu_scene);

        self.blit_bind_group_layout =
            Some(device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("blit-layout"),
                bindings: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::SampledTexture {
                            multisampled: false,
                            component_type: wgpu::TextureComponentType::Uint,
                            dimension: wgpu::TextureViewDimension::D2,
                        },
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler { comparison: false },
                    },
                ],
            }));

        self.output_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1.0,
            compare: wgpu::CompareFunction::Never,
        }));

        self.resize(width, height, device, _requests);

        let blit_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: &[self.blit_bind_group_layout.as_ref().unwrap()],
        });

        let vert_shader = "shaders/quad.vert";
        let frag_shader = "shaders/quad.frag";

        let vert_module = self
            .compiler
            .compile_from_file(vert_shader, ShaderKind::Vertex)
            .unwrap();
        let frag_module = self
            .compiler
            .compile_from_file(frag_shader, ShaderKind::Fragment)
            .unwrap();

        let vert_module = device.create_shader_module(vert_module.as_slice());
        let frag_module = device.create_shader_module(frag_module.as_slice());

        self.blit_pipeline = Some(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                layout: &blit_pipeline_layout,
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
                    cull_mode: wgpu::CullMode::None,
                    depth_bias: 0,
                    depth_bias_slope_scale: 0.0,
                    depth_bias_clamp: 0.0,
                }),
                primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                color_states: &[wgpu::ColorStateDescriptor {
                    format: self.sc_format,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                }],
                depth_stencil_state: None,
                vertex_state: wgpu::VertexStateDescriptor {
                    index_format: wgpu::IndexFormat::Uint32,
                    vertex_buffers: &[],
                },
                sample_count: 1,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            }),
        );
    }

    fn render(
        &mut self,
        fb: &wgpu::SwapChainOutput,
        device: &wgpu::Device,
        requests: &mut VecDeque<Request>,
    ) {
        use wgpu::*;
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("render-encoder"),
        });

        if let Some(scene) = self.gpu_scene.as_ref() {
            scene.record_render(
                &self.camera,
                device,
                &mut encoder,
                self.output_texture_view.as_ref().unwrap(),
                self.depth_texture_view.as_ref().unwrap(),
            );
        }

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[RenderPassColorAttachmentDescriptor {
                    attachment: &fb.view,
                    clear_color: Color::BLACK,
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    resolve_target: None,
                }],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(self.blit_pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, self.blit_bind_group.as_ref().unwrap(), &[]);
            render_pass.draw(0..6, 0..1);
        }

        requests.push_back(Request::CommandBuffer(encoder.finish()));
    }

    fn mouse_button_handling(
        &mut self,
        _states: &MouseButtonHandler,
        _requests: &mut VecDeque<Request>,
    ) {
    }

    fn key_handling(&mut self, states: &KeyHandler, requests: &mut VecDeque<Request>) {
        #[cfg(target_os = "macos")]
        {
            if states.pressed(KeyCode::LWin) && states.pressed(KeyCode::Q) {
                requests.push_back(Request::Exit);
                return;
            }
        }

        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            if states.pressed(KeyCode::LAlt) && states.pressed(KeyCode::F4) {
                requests.push_back(Request::Exit);
                return;
            }
        }

        if states.pressed(KeyCode::Escape) {
            requests.push_back(Request::Exit);
            return;
        }

        let mut view_change = Vec3::new(0.0, 0.0, 0.0);
        let mut pos_change = Vec3::new(0.0, 0.0, 0.0);

        if states.pressed(KeyCode::Up) {
            view_change += (0.0, 1.0, 0.0).into();
        }
        if states.pressed(KeyCode::Down) {
            view_change -= (0.0, 1.0, 0.0).into();
        }
        if states.pressed(KeyCode::Left) {
            view_change -= (1.0, 0.0, 0.0).into();
        }
        if states.pressed(KeyCode::Right) {
            view_change += (1.0, 0.0, 0.0).into();
        }

        if states.pressed(KeyCode::W) {
            pos_change += (0.0, 0.0, 1.0).into();
        }
        if states.pressed(KeyCode::S) {
            pos_change -= (0.0, 0.0, 1.0).into();
        }
        if states.pressed(KeyCode::A) {
            pos_change -= (1.0, 0.0, 0.0).into();
        }
        if states.pressed(KeyCode::D) {
            pos_change += (1.0, 0.0, 0.0).into();
        }
        if states.pressed(KeyCode::E) {
            pos_change += (0.0, 1.0, 0.0).into();
        }
        if states.pressed(KeyCode::Q) {
            pos_change -= (0.0, 1.0, 0.0).into();
        }

        let elapsed = self.timer.elapsed_in_millis();
        let elapsed = if states.pressed(KeyCode::LShift) {
            elapsed * 2.0
        } else {
            elapsed
        };

        let view_change = view_change * elapsed * 0.001;
        let pos_change = pos_change * elapsed * 0.01;

        let mut camera = &mut self.camera;

        if view_change != [0.0; 3].into() {
            camera.translate_target(view_change);
        }
        if pos_change != [0.0; 3].into() {
            camera.translate_relative(pos_change);
        }

        let elapsed = self.timer.elapsed_in_millis();

        if states.pressed(KeyCode::RBracket) {
            camera.speed += elapsed / 10.0;
        }
        if states.pressed(KeyCode::LBracket) {
            camera.speed -= elapsed / 10.0;
        }
        camera.speed = camera.speed.max(0.1);

        self.frame_time.add_sample(elapsed);
        self.fps.add_sample(1000.0 / elapsed);
        let frame_time_avg = self.frame_time.get_average();
        let fps_avg = self.fps.get_average();
        self.timer.reset();

        requests.push_back(Request::TitleChange(format!(
            "Frame time: {:.2} ms, FPS: {:.2}",
            frame_time_avg, fps_avg
        )))
    }

    fn mouse_handling(
        &mut self,
        _x: f64,
        _y: f64,
        _delta_x: f64,
        _delta_y: f64,
        _requests: &mut VecDeque<Request>,
    ) {
    }

    fn scroll_handling(&mut self, _dx: f64, dy: f64, _requests: &mut VecDeque<Request>) {
        self.camera
            .change_fov(self.camera.get_fov() - (dy as f32) * 0.01);
    }

    fn resize(
        &mut self,
        width: u32,
        height: u32,
        device: &wgpu::Device,
        _requests: &mut VecDeque<Request>,
    ) {
        use wgpu::*;

        self.width = width;
        self.height = height;
        self.camera.resize(width, height);

        let new_texture = device.create_texture(&TextureDescriptor {
            label: Some("output-texture"),
            size: Extent3d {
                width: self.width,
                height: self.height,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: Self::OUTPUT_FORMAT,
            usage: TextureUsage::OUTPUT_ATTACHMENT | TextureUsage::SAMPLED,
        });
        let new_texture_view = new_texture.create_default_view();

        self.output_texture_view = Some(new_texture_view);
        self.output_texture = Some(new_texture);

        self.blit_bind_group = Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("blit-bind-group"),
            bindings: &[
                Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        self.output_texture_view.as_ref().unwrap(),
                    ),
                },
                Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(self.output_sampler.as_ref().unwrap()),
                },
            ],
            layout: self.blit_bind_group_layout.as_ref().unwrap(),
        }));

        let new_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth-texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        });
        let new_view = new_texture.create_default_view();

        self.depth_texture_view = Some(new_view);
        self.depth_texture = Some(new_texture);
    }
}
