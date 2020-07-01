use ash::extensions::{
    ext::DebugUtils,
    khr::{Surface, Swapchain},
};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::{vk, Entry};
use scene::renderers::{RenderMode, Renderer, Setting};
use scene::{
    raw_window_handle::HasRawWindowHandle, AreaLight, BitVec, Camera, DeviceMaterial,
    DirectionalLight, Instance, Local, Material, Mesh, PointLight, SpotLight, Texture,
};
use shared::*;
use std::error::Error;
use std::{
    ffi::{CStr, CString},
    fmt::Display,
};

use super::helpers::*;

#[derive(Debug, Copy, Clone)]
pub enum RendererError {
    Unknown,
}

impl Error for RendererError {}

impl Display for RendererError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

pub struct VkRenderer<'a> {
    compiler: Compiler<'a>,
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub surface_loader: Surface,
    pub swapchain_loader: Swapchain,
    pub debug_utils_loader: DebugUtils,
    pub debug_call_back: vk::DebugUtilsMessengerEXT,
    pub pdevice: vk::PhysicalDevice,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_index: u32,
    pub present_queue: vk::Queue,

    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,

    pub swapchain: vk::SwapchainKHR,
    pub swapchain_create_info: vk::SwapchainCreateInfoKHR,
    pub present_images: Vec<vk::Image>,
    pub present_image_views: Vec<vk::ImageView>,

    pub pool: vk::CommandPool,
    pub draw_command_buffer: vk::CommandBuffer,
    pub setup_command_buffer: vk::CommandBuffer,

    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub depth_image_memory: vk::DeviceMemory,

    pub present_complete_semaphore: vk::Semaphore,
    pub rendering_complete_semaphore: vk::Semaphore,

    pub pipeline_cache: vk::PipelineCache,
    pub debug_pipeline_layout: vk::PipelineLayout,
    pub debug_pipeline: vk::Pipeline,
    pub render_pass: vk::RenderPass,
}

impl<'a> VkRenderer<'a> {
    pub const DEPTH_FORMAT: vk::Format = vk::Format::D16_UNORM;
}

impl<'a> Renderer for VkRenderer<'a> {
    fn init<T: HasRawWindowHandle>(
        window: &T,
        width: usize,
        height: usize,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        unsafe {
            let entry = Entry::new().unwrap();
            let app_name = CString::new("Vulkan Renderer").unwrap();
            let engine_name = CString::new("rfw-rs").unwrap();

            let layer_names = [CString::new("VK_LAYER_KHRONOS_validation")?];
            let layers_names_raw: Vec<*const i8> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let mut extension_names: Vec<&CStr> =
                ash_window::enumerate_required_extensions(window).unwrap();
            extension_names.push(ash::extensions::ext::DebugUtils::name());

            let extension_names_raw: Vec<*const i8> = extension_names
                .iter()
                .map(|s| {
                    println!("{}", s.to_str().unwrap());
                    s.as_ptr()
                })
                .collect();

            let app_info = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .application_version(0)
                .engine_name(&engine_name)
                .engine_version(0)
                .api_version(vk::make_version(1, 1, 0))
                .build();

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_layer_names(&layers_names_raw)
                .enabled_extension_names(&extension_names_raw);
            let instance: ash::Instance = entry.create_instance(&create_info, None).unwrap();
            println!("Created instance");

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            let debug_call_back = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap();

            let surface: vk::SurfaceKHR =
                ash_window::create_surface(&entry, &instance, window, None).unwrap();
            let pdevices = instance
                .enumerate_physical_devices()
                .expect("Could not retrieve physical devices");
            let surface_loader = Surface::new(&entry, &instance);
            let (pdevice, queue_family_index) = pdevices
                .iter()
                .map(|pdevice| {
                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .filter_map(|(index, ref info)| {
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader
                                        .get_physical_device_surface_support(
                                            *pdevice,
                                            index as u32,
                                            surface,
                                        )
                                        .unwrap();
                            if supports_graphic_and_surface {
                                Some((*pdevice, index))
                            } else {
                                None
                            }
                        })
                        .next()
                })
                .filter_map(|v| v)
                .next()
                .expect("Could not find suitable device");

            let queue_family_index = queue_family_index as u32;
            let device_extension_names_raw = [Swapchain::name().as_ptr()];
            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                ..Default::default()
            };
            let priorities = [1.0];

            let queue_info = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)
                .build()];

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&device_extension_names_raw)
                .enabled_features(&features);

            let props = instance.get_physical_device_properties(pdevice);
            let device_name = CStr::from_ptr(props.device_name.as_ptr());
            println!(
                "Picked device: {}",
                CString::from(device_name).to_str().unwrap()
            );

            // return Err(Box::new(RendererError::Unknown));
            let device = instance
                .create_device(pdevice, &device_create_info, None)
                .unwrap();
            println!("Created device.");

            let present_queue = device.get_device_queue(queue_family_index as u32, 0);

            let surface_formats = surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap();

            let surface_format = surface_formats
                .iter()
                .map(|sfmt| match sfmt.format {
                    vk::Format::UNDEFINED => vk::SurfaceFormatKHR {
                        format: vk::Format::B8G8R8_UNORM,
                        color_space: sfmt.color_space,
                    },
                    _ => *sfmt,
                })
                .next()
                .expect("Could not find suitable surface format");

            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(pdevice, surface)
                .unwrap();
            let mut desired_image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }
            let surface_resolution = match surface_capabilities.current_extent.width {
                std::u32::MAX => vk::Extent2D {
                    width: width as u32,
                    height: height as u32,
                },
                _ => surface_capabilities.current_extent,
            };
            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };
            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(pdevice, surface)
                .unwrap();
            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::FIFO)
                .unwrap_or(vk::PresentModeKHR::FIFO);
            let swapchain_loader = Swapchain::new(&instance, &device);

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_resolution)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1)
                .build();

            let swapchain = swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();
            println!("Created swapchain");

            let pool_create_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);

            let pool = device.create_command_pool(&pool_create_info, None).unwrap();

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(2)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers = device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap();
            let setup_command_buffer = command_buffers[0];
            let draw_command_buffer = command_buffers[1];

            let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
            let present_image_views: Vec<vk::ImageView> = present_images
                .iter()
                .map(|&image| {
                    let create_view_info = vk::ImageViewCreateInfo::builder()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(image);
                    device.create_image_view(&create_view_info, None).unwrap()
                })
                .collect();

            let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);
            let depth_image_create_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(Self::DEPTH_FORMAT)
                .extent(vk::Extent3D {
                    width: surface_resolution.width,
                    height: surface_resolution.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let depth_image = device.create_image(&depth_image_create_info, None).unwrap();
            let depth_image_memory_req = device.get_image_memory_requirements(depth_image);
            let depth_image_memory_index = find_memorytype_index(
                &depth_image_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Unable to find suitable memory index for depth image.");

            let depth_image_allocate_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(depth_image_memory_req.size)
                .memory_type_index(depth_image_memory_index)
                .build();

            let depth_image_memory = device
                .allocate_memory(&depth_image_allocate_info, None)
                .unwrap();

            device
                .bind_image_memory(depth_image, depth_image_memory, 0)
                .unwrap();

            let depth_image_view_info = vk::ImageViewCreateInfo::builder()
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1)
                        .build(),
                )
                .image(depth_image)
                .format(depth_image_create_info.format)
                .view_type(vk::ImageViewType::TYPE_2D);

            let depth_image_view = device
                .create_image_view(&depth_image_view_info, None)
                .unwrap();

            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let present_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            let rendering_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            let debug_pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo {
                        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        set_layout_count: 0,
                        p_set_layouts: std::ptr::null(),
                        push_constant_range_count: 0,
                        p_push_constant_ranges: std::ptr::null(),
                    },
                    None,
                )
                .unwrap();
            let quad_vert = include_str!("../shaders/quad.vert");
            let quad_frag = include_str!("../shaders/quad.frag");

            let mut compiler = CompilerBuilder::new()
                .with_opt_level(OptimizationLevel::Performance)
                .build()
                .unwrap();

            let pipeline_cache = device
                .create_pipeline_cache(
                    &vk::PipelineCacheCreateInfo {
                        s_type: vk::StructureType::PIPELINE_CACHE_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        initial_data_size: 0,
                        p_initial_data: std::ptr::null(),
                    },
                    None,
                )
                .unwrap();

            let vert_module = compiler
                .compile_from_string(quad_vert, ShaderKind::Vertex)
                .unwrap();
            let frag_module = compiler
                .compile_from_string(quad_frag, ShaderKind::Fragment)
                .unwrap();
            let vert_module = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        code_size: vert_module.len() * std::mem::size_of::<u32>(),
                        p_code: vert_module.as_ptr(),
                    },
                    None,
                )
                .unwrap();
            let frag_module = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        code_size: frag_module.len() * std::mem::size_of::<u32>(),
                        p_code: frag_module.as_ptr(),
                    },
                    None,
                )
                .unwrap();

            let stages = [
                vk::PipelineShaderStageCreateInfo {
                    s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: Default::default(),
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: vert_module,
                    p_name: CString::new("main").unwrap().as_ptr(),
                    p_specialization_info: std::ptr::null(),
                },
                vk::PipelineShaderStageCreateInfo {
                    s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: Default::default(),
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    module: frag_module,
                    p_name: CString::new("main").unwrap().as_ptr(),
                    p_specialization_info: std::ptr::null(),
                },
            ];

            let attachment_info = [
                vk::AttachmentDescription {
                    flags: Default::default(),
                    format: surface_format.format,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE,
                    stencil_load_op: vk::AttachmentLoadOp::CLEAR,
                    stencil_store_op: vk::AttachmentStoreOp::STORE,
                    initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    final_layout: vk::ImageLayout::GENERAL,
                },
                vk::AttachmentDescription {
                    flags: Default::default(),
                    format: Self::DEPTH_FORMAT,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE,
                    stencil_load_op: vk::AttachmentLoadOp::CLEAR,
                    stencil_store_op: vk::AttachmentStoreOp::STORE,
                    initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    final_layout: vk::ImageLayout::GENERAL,
                },
            ];

            let sub_pass = vk::SubpassDescription::builder()
                .color_attachments(&[vk::AttachmentReference::builder()
                    .attachment(0)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .build()])
                .depth_stencil_attachment(&vk::AttachmentReference::builder().attachment(1))
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .build();
            let render_pass = vk::RenderPassCreateInfo {
                s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: Default::default(),
                attachment_count: 1,
                p_attachments: attachment_info.as_ptr(),
                subpass_count: 1,
                p_subpasses: &sub_pass,
                dependency_count: 0,
                p_dependencies: std::ptr::null(),
            };

            let render_pass = device.create_render_pass(&render_pass, None).unwrap();
            let scissor = vk::Rect2D::builder()
                .extent(vk::Extent2D {
                    width: width as u32,
                    height: height as u32,
                })
                .offset(vk::Offset2D { x: 0, y: 0 })
                .build();

            let debug_create_info = vk::GraphicsPipelineCreateInfo {
                s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: Default::default(),
                stage_count: 2,
                p_stages: stages.as_ptr(),
                p_vertex_input_state: &vk::PipelineVertexInputStateCreateInfo {
                    s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: Default::default(),
                    vertex_binding_description_count: 0,
                    p_vertex_binding_descriptions: std::ptr::null(),
                    vertex_attribute_description_count: 0,
                    p_vertex_attribute_descriptions: std::ptr::null(),
                },
                p_input_assembly_state: std::ptr::null(),
                p_tessellation_state: std::ptr::null(),
                p_viewport_state: &vk::PipelineViewportStateCreateInfo {
                    s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: Default::default(),
                    viewport_count: 1,
                    p_viewports: &vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: width as f32,
                        height: height as f32,
                        min_depth: 0.0,
                        max_depth: 0.0,
                    },
                    scissor_count: 1,
                    p_scissors: &scissor,
                },
                p_rasterization_state: &vk::PipelineRasterizationStateCreateInfo {
                    s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: Default::default(),
                    depth_clamp_enable: 0,
                    rasterizer_discard_enable: 0,
                    polygon_mode: vk::PolygonMode::FILL,
                    cull_mode: vk::CullModeFlags::NONE,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    depth_bias_enable: 0,
                    depth_bias_constant_factor: 0.0,
                    depth_bias_clamp: 0.0,
                    depth_bias_slope_factor: 0.0,
                    line_width: 0.0,
                },
                p_multisample_state: &vk::PipelineMultisampleStateCreateInfo {
                    s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: Default::default(),
                    rasterization_samples: vk::SampleCountFlags::TYPE_1,
                    sample_shading_enable: 0,
                    min_sample_shading: 0.0,
                    p_sample_mask: [!0 as vk::SampleMask].as_ptr(),
                    alpha_to_coverage_enable: 0,
                    alpha_to_one_enable: 0,
                },
                p_depth_stencil_state: &vk::PipelineDepthStencilStateCreateInfo {
                    s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: Default::default(),
                    depth_test_enable: 0,
                    depth_write_enable: 0,
                    depth_compare_op: Default::default(),
                    depth_bounds_test_enable: 0,
                    stencil_test_enable: 0,
                    front: Default::default(),
                    back: Default::default(),
                    min_depth_bounds: 0.0,
                    max_depth_bounds: 0.0,
                },
                p_color_blend_state: &vk::PipelineColorBlendStateCreateInfo {
                    s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: Default::default(),
                    logic_op_enable: 0,
                    logic_op: Default::default(),
                    attachment_count: 0,
                    p_attachments: [vk::PipelineColorBlendAttachmentState {
                        blend_enable: 0,
                        src_color_blend_factor: Default::default(),
                        dst_color_blend_factor: Default::default(),
                        color_blend_op: Default::default(),
                        src_alpha_blend_factor: Default::default(),
                        dst_alpha_blend_factor: Default::default(),
                        alpha_blend_op: Default::default(),
                        color_write_mask: Default::default(),
                    }]
                    .as_ptr(),
                    blend_constants: [1.0; 4],
                },
                p_dynamic_state: std::ptr::null(),
                layout: debug_pipeline_layout,
                render_pass: render_pass,
                subpass: 1,
                base_pipeline_handle: Default::default(),
                base_pipeline_index: 0,
            };

            let debug_pipeline = device
                .create_graphics_pipelines(pipeline_cache, &[debug_create_info], None)
                .unwrap()[0];

            println!("Initialized");

            Ok(Box::new(Self {
                compiler,
                entry,
                instance,
                device,
                surface_loader,
                swapchain_loader,
                debug_utils_loader,
                debug_call_back,
                pdevice,
                device_memory_properties,
                queue_family_index,
                present_queue,
                surface,
                surface_format,
                surface_resolution,
                swapchain,
                swapchain_create_info,
                present_images,
                present_image_views,
                pool,
                draw_command_buffer,
                setup_command_buffer,
                depth_image,
                depth_image_view,
                depth_image_memory,
                present_complete_semaphore,
                rendering_complete_semaphore,
                pipeline_cache,
                debug_pipeline_layout,
                debug_pipeline,
                render_pass,
            }))
        }
    }

    fn set_mesh(&mut self, _id: usize, _mesh: &Mesh) {}

    fn set_instance(&mut self, _id: usize, _instance: &Instance) {}

    fn set_materials(&mut self, _materials: &[Material], _device_materials: &[DeviceMaterial]) {}

    fn set_textures(&mut self, _textures: &[Texture]) {}

    fn synchronize(&mut self) {}

    fn render(&mut self, _camera: &Camera, _mode: RenderMode) {
        record_submit_commandbuffer(
            &self.device,
            self.setup_command_buffer,
            self.present_queue,
            &[],
            &[],
            &[],
            |device, _command_buffer| {
                let layout_transition_barriers = vk::ImageMemoryBarrier::builder()
                    .image(self.depth_image)
                    .dst_access_mask(
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    )
                    .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .layer_count(1)
                            .level_count(1)
                            .build(),
                    );

                unsafe {
                    device.cmd_pipeline_barrier(
                        self.setup_command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[layout_transition_barriers.build()],
                    );
                }
            },
        );
    }

    fn resize<T: HasRawWindowHandle>(&mut self, _window: &T, width: usize, height: usize) {
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        unsafe {
            self.present_complete_semaphore = self
                .device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            self.rendering_complete_semaphore = self
                .device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            self.surface_resolution.width = width as u32;
            self.surface_resolution.height = height as u32;

            self.swapchain_create_info.image_extent.width = width as u32;
            self.swapchain_create_info.image_extent.height = height as u32;
            self.swapchain_create_info.old_swapchain = self.swapchain;
            self.swapchain = self
                .swapchain_loader
                .create_swapchain(&self.swapchain_create_info, None)
                .unwrap();

            self.present_images = self
                .swapchain_loader
                .get_swapchain_images(self.swapchain)
                .unwrap();
            self.present_image_views = self
                .present_images
                .iter()
                .map(|&image| {
                    let create_view_info = vk::ImageViewCreateInfo::builder()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(image);
                    self.device
                        .create_image_view(&create_view_info, None)
                        .unwrap()
                })
                .collect();

            let depth_image_create_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(Self::DEPTH_FORMAT)
                .extent(vk::Extent3D {
                    width: self.surface_resolution.width,
                    height: self.surface_resolution.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            self.depth_image = self
                .device
                .create_image(&depth_image_create_info, None)
                .unwrap();
            let depth_image_memory_req =
                self.device.get_image_memory_requirements(self.depth_image);
            let depth_image_memory_index = find_memorytype_index(
                &depth_image_memory_req,
                &self.device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Unable to find suitable memory index for depth image.");

            let depth_image_allocate_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(depth_image_memory_req.size)
                .memory_type_index(depth_image_memory_index)
                .build();

            self.depth_image_memory = self
                .device
                .allocate_memory(&depth_image_allocate_info, None)
                .unwrap();

            self.device
                .bind_image_memory(self.depth_image, self.depth_image_memory, 0)
                .unwrap();

            let depth_image_view_info = vk::ImageViewCreateInfo::builder()
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1)
                        .build(),
                )
                .image(self.depth_image)
                .format(depth_image_create_info.format)
                .view_type(vk::ImageViewType::TYPE_2D);

            self.depth_image_view = self
                .device
                .create_image_view(&depth_image_view_info, None)
                .unwrap();
        }

        println!("Recreated swapchain");
    }

    fn set_point_lights(&mut self, _changed: &BitVec<Local, usize>, _lights: &[PointLight]) {}

    fn set_spot_lights(&mut self, _changed: &BitVec<Local, usize>, _lights: &[SpotLight]) {}

    fn set_area_lights(&mut self, _changed: &BitVec<Local, usize>, _lights: &[AreaLight]) {}

    fn set_directional_lights(
        &mut self,
        _changed: &BitVec<Local, usize>,
        _lights: &[DirectionalLight],
    ) {
    }

    fn get_settings(&self) -> Vec<Setting> {
        Vec::new()
    }

    fn set_setting(&mut self, _setting: Setting) {}
}