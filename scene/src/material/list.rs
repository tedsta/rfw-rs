use crate::material::Material;

use glam::*;
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use image::GenericImageView;
use rayon::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialList {
    materials: Vec<Material>,
    tex_path_mapping: HashMap<PathBuf, usize>,
    textures: Vec<Texture>,
}


// TODO: Support other formats than BGRA8
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Texture {
    pub data: Vec<u32>,
    pub width: u32,
    pub height: u32,
}

impl Texture {
    pub fn generate_mipmaps(&mut self, levels: usize) {
        self.data.resize(self.required_texel_count(levels), 0);

        let mut src_offset = 0;
        let mut dst_offset = src_offset + self.width as usize * self.height as usize;

        let mut pw = self.width as usize;
        let mut w = self.width as usize >> 1;
        let mut h = self.height as usize >> 1;

        for _ in 1..levels {
            let max_dst_offset = dst_offset + (w * h);
            assert!(max_dst_offset <= self.data.len());

            for y in 0..h {
                for x in 0..w {
                    let src0 = self.data[x * 2 + (y * 2) * pw + src_offset];
                    let src1 = self.data[x * 2 + 1 + (y * 2) * pw + src_offset];
                    let src2 = self.data[x * 2 + (y * 2 + 1) * pw + src_offset];
                    let src3 = self.data[x * 2 + 1 + (y * 2 + 1) * pw + src_offset];
                    let a = ((src0 >> 24) & 255).min((src1 >> 24) & 255).min(
                        ((src2 >> 24) & 255).min((src3 >> 24) & 255));
                    let r =
                        ((src0 >> 16) & 255) + ((src1 >> 16) & 255) + ((src2 >> 16) & 255) + ((src3 >> 16) & 255);
                    let g =
                        ((src0 >> 8) & 255) + ((src1 >> 8) & 255) + ((src2 >> 8) & 255) + ((src3 >> 8) & 255);
                    let b = (src0 & 255) + (src1 & 255) + (src2 & 255) + (src3 & 255);
                    self.data[dst_offset + x + y * w] = (a << 24) + ((r >> 2) << 16) + ((g >> 2) << 8) + (b >> 2);
                }
            }

            src_offset = dst_offset;
            dst_offset += w * h;
            pw = w;
            w >>= 1;
            h >>= 1;
        }
    }

    fn required_texel_count(&self, levels: usize) -> usize {
        let mut w = self.width;
        let mut h = self.height;
        let mut needed = 0;

        for _ in 0..levels {
            needed += w * h;
            w >>= 1;
            h >>= 1;
        }

        needed as usize
    }
}

#[allow(dead_code)]
impl MaterialList {
    pub fn new() -> MaterialList {
        let materials = vec![Material::new(
            vec3(1.0, 0.0, 0.0),
            1.0,
            vec3(1.0, 0.0, 0.0),
            1.0,
        )];

        let mut textures = Vec::new();
        textures.push(Texture { // Make sure always a single texture exists (as fallback)
            width: 64,
            height: 64,
            data: vec![0; 4096],
        });

        MaterialList {
            materials,
            tex_path_mapping: HashMap::new(),
            textures,
        }
    }

    pub fn add(&mut self, color: Vec3, roughness: f32, specular: Vec3, opacity: f32) -> usize {
        let material = Material::new(color, roughness, specular, opacity);
        self.push(material)
    }

    pub fn add_with_maps<T: AsRef<Path> + Copy>(&mut self, color: Vec3, roughness: f32, specular: Vec3, opacity: f32, albedo: Option<T>, normal: Option<T>) -> usize {
        let mut material = Material::new(color, roughness, specular, opacity);
        let diffuse_tex = if let Some(path) = albedo {
            match self.get_texture_index(path) {
                Ok(id) => id,
                _ => -1
            }
        } else { -1 };

        let normal_tex = if let Some(path) = normal {
            match self.get_texture_index(path) {
                Ok(id) => id,
                _ => -1
            }
        } else { -1 };

        material.diffuse_tex = diffuse_tex;
        material.normal_tex = normal_tex;
        self.push(material)
    }

    pub fn push(&mut self, mat: Material) -> usize {
        let i = self.materials.len();
        self.materials.push(mat);
        i
    }

    pub fn get(&self, index: usize) -> Option<&Material> {
        self.materials.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut Material> {
        self.materials.get_mut(index)
    }

    pub unsafe fn get_unchecked(&self, index: usize) -> &Material {
        self.materials.get_unchecked(index)
    }

    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &Material {
        self.materials.get_unchecked_mut(index)
    }

    pub fn get_texture_index<T: AsRef<Path> + Copy>(&mut self, path: T) -> Result<i32, i32> {
        // First see if we have already loaded the texture before
        if let Some(id) = self.tex_path_mapping.get(path.as_ref()) {
            return Ok((*id) as i32);
        }

        // See if file exists
        if !path.as_ref().exists() {
            return Err(-1);
        }

        // Attempt to load image
        let img = image::open(path);
        if let Err(_) = img {
            return Err(-1);
        }

        // Loading was successful
        let img = img.unwrap().rotate180();

        let (width, height) = (img.width(), img.height());
        let mut data = vec![0 as u32; (width * height) as usize];

        let bgra_image = img.to_bgra();
        data.copy_from_slice(unsafe {
            std::slice::from_raw_parts(bgra_image.as_ptr() as *const u32, (width * height) as usize)
        });

        let tex = Texture {
            width,
            height,
            data,
        };

        self.textures.push(tex);
        let index = self.textures.len() - 1;

        // Add to mapping to prevent loading the same image multiple times
        self.tex_path_mapping.insert(path.as_ref().to_path_buf(), index);

        Ok(index as i32)
    }

    pub fn get_default(&self) -> usize {
        0
    }

    #[cfg(feature = "wgpu")]
    pub fn create_wgpu_buffer(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::BufferAddress, wgpu::Buffer) {
        use wgpu::*;

        let size = (self.materials.len() * std::mem::size_of::<Material>()) as BufferAddress;
        let staging_buffer = device.create_buffer_mapped(&wgpu::BufferDescriptor {
            label: Some("material-staging-buffer"),
            size,
            usage: BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC,
        });
        staging_buffer.data.copy_from_slice(unsafe {
            std::slice::from_raw_parts(self.materials.as_ptr() as *const u8, size as usize)
        });
        let staging_buffer = staging_buffer.finish();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material-buffer"),
            size,
            usage: BufferUsage::COPY_DST | BufferUsage::STORAGE_READ,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("create-material-cmd-buffer")
        });

        encoder.copy_buffer_to_buffer(&staging_buffer, 0, &buffer, 0, size);
        queue.submit(&[encoder.finish()]);

        (size, buffer)
    }

    #[cfg(feature = "wgpu")]
    pub fn create_wgpu_textures(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<wgpu::Texture> {
        use wgpu::*;

        const MIP_LEVELS: usize = 5;

        self.textures.par_iter_mut().for_each(|tex| { tex.generate_mipmaps(MIP_LEVELS); });
        let mut textures = Vec::new();

        let staging_size = self.textures.iter().map(|t| t.data.len()).sum::<usize>() * std::mem::size_of::<u32>();
        let staging_buffer = device.create_buffer_mapped(&wgpu::BufferDescriptor {
            label: Some("material-staging-buffer"),
            size: staging_size as BufferAddress,
            usage: BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC,
        });

        let mut data_ptr = staging_buffer.data.as_mut_ptr() as *mut u32;
        for tex in self.textures.iter() {
            unsafe {
                std::ptr::copy(tex.data.as_ptr(), data_ptr, tex.data.len());
                data_ptr = data_ptr.add(tex.data.len());
            }
        }
        let staging_buffer = staging_buffer.finish();

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("texture-staging-cmd-buffer")
        });

        let mut offset = 0 as BufferAddress;
        for (i, tex) in self.textures.iter_mut().enumerate() {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some(format!("texture-{}", i).as_str()),
                size: Extent3d { width: tex.width, height: tex.height, depth: 1 },
                array_layer_count: 1,
                mip_level_count: 5,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Bgra8Unorm,
                usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST,
            });

            let mut width = tex.width;
            let mut height = tex.height;
            let mut local_offset = 0 as BufferAddress;
            for i in 0..MIP_LEVELS {
                encoder.copy_buffer_to_texture(BufferCopyView {
                    buffer: &staging_buffer,
                    offset: offset + local_offset * std::mem::size_of::<u32>() as BufferAddress,
                    bytes_per_row: (width as usize * std::mem::size_of::<u32>()) as u32,
                    rows_per_image: tex.height,
                }, TextureCopyView {
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    array_layer: 0,
                    mip_level: i as u32,
                    texture: &texture,
                }, Extent3d { width, height, depth: 1 },
                );

                local_offset += (width * height) as BufferAddress;
                width >>= 1;
                height >>= 1;
            }

            offset += (tex.data.len() * std::mem::size_of::<u32>()) as BufferAddress;
            textures.push(texture);
        }

        queue.submit(&[encoder.finish()]);

        textures
    }

    pub fn len(&self) -> usize {
        self.materials.len()
    }
}

impl Index<usize> for MaterialList {
    type Output = Material;

    fn index(&self, index: usize) -> &Self::Output {
        &self.materials[index]
    }
}

impl IndexMut<usize> for MaterialList {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.materials[index]
    }
}
