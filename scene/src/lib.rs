pub mod camera;
pub mod constants;
pub mod graph;
pub mod intersector;
pub mod lights;
pub mod loaders;
pub mod material;
pub mod objects;
pub mod renderers;
pub mod scene;
pub mod triangle_scene;

pub use specs::prelude::*;

mod utils;

pub use camera::*;
pub use intersector::*;
pub use lights::*;
pub use loaders::*;
pub use material::*;
pub use objects::*;
pub use scene::*;

use renderers::{RenderMode, Renderer, Setting};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

pub use bitvec::prelude::*;
pub use raw_window_handle;
pub use triangle_scene::*;

use glam::*;
use std::error::Error;
use std::path::Path;

pub struct InstanceRef {
    id: usize,
    translation: Vec3,
    scaling: Vec3,
    rotation_x: f32,
    rotation_y: f32,
    rotation_z: f32,
    changed: bool,
}

impl Component for InstanceRef {
    type Storage = FlaggedStorage<Self>;
}

#[allow(dead_code)]
impl InstanceRef {
    fn new(id: usize) -> InstanceRef {
        Self {
            id,
            translation: Vec3::zero(),
            scaling: Vec3::one(),
            rotation_x: 0.0,
            rotation_y: 0.0,
            rotation_z: 0.0,
            changed: true,
        }
    }

    pub fn translate_x(&mut self, offset: f32) {
        self.translation += Vec3::new(offset, 0.0, 0.0);
        self.changed = true;
    }

    pub fn translate_y(&mut self, offset: f32) {
        self.translation += Vec3::new(0.0, offset, 0.0);
        self.changed = true;
    }

    pub fn translate_z(&mut self, offset: f32) {
        self.translation += Vec3::new(0.0, 0.0, offset);
        self.changed = true;
    }

    pub fn rotate_x(&mut self, degrees: f32) {
        self.rotation_x = (self.rotation_x + degrees.to_radians()) % (std::f32::consts::PI * 2.0);
        self.changed = true;
    }

    pub fn rotate_y(&mut self, degrees: f32) {
        self.rotation_y = (self.rotation_y + degrees.to_radians()) % (std::f32::consts::PI * 2.0);
        self.changed = true;
    }

    pub fn rotate_z(&mut self, degrees: f32) {
        self.rotation_z = (self.rotation_z + degrees.to_radians()) % (std::f32::consts::PI * 2.0);
        self.changed = true;
    }

    pub fn scale<T: Into<[f32; 3]>>(&mut self, scale: T) {
        let scale: [f32; 3] = scale.into();
        let scale: Vec3 = Vec3::from(scale).max(Vec3::splat(0.001));
        self.scaling *= scale;
        self.changed = true;
    }

    pub fn scale_x(&mut self, scale: f32) {
        let scale = scale.max(0.001);
        self.scaling[0] *= scale;
        self.changed = true;
    }

    pub fn scale_y(&mut self, scale: f32) {
        let scale = scale.max(0.001);
        self.scaling[1] *= scale;
        self.changed = true;
    }

    pub fn scale_z(&mut self, scale: f32) {
        let scale = scale.max(0.001);
        self.scaling[2] *= scale;
        self.changed = true;
    }

    /// Returns translation in [x, y, z]
    pub fn get_translation(&self) -> [f32; 3] {
        self.translation.into()
    }

    /// Returns scale in [x, y, z]
    pub fn get_scale(&self) -> [f32; 3] {
        self.scaling.into()
    }

    /// Returns rotation as quaternion in [x, y, z, w]
    pub fn get_rotation(&self) -> [f32; 4] {
        let mut quat = Quat::identity();
        if self.rotation_x.abs() > 0.0001 {
            quat *= Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), self.rotation_x);
        }
        if self.rotation_y.abs() > 0.0001 {
            quat *= Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), self.rotation_y);
        }
        if self.rotation_z.abs() > 0.0001 {
            quat *= Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), self.rotation_z);
        }

        quat.into()
    }

    /// Returns rotation as radian euler angles in [x, y, z]
    pub fn get_euler_angles(&self) -> [f32; 3] {
        [self.rotation_x, self.rotation_y, self.rotation_z]
    }

    pub fn trs_matrix(&self) -> Mat4 {
        let rotation_x = Quat::from_axis_angle(Vec3::unit_x(), self.rotation_x);
        let rotation_y = Quat::from_axis_angle(Vec3::unit_y(), self.rotation_y);
        let rotation_z = Quat::from_axis_angle(Vec3::unit_z(), self.rotation_z);
        Mat4::from_scale_rotation_translation(
            self.scaling.into(),
            rotation_x * rotation_y * rotation_z,
            self.translation.into(),
        )
    }
}

#[derive(Debug, Clone)]
pub enum SceneLight {
    Point(PointLight),
    Spot(SpotLight),
    Directional(DirectionalLight),
}

pub struct LightRef {
    id: usize,
    light: SceneLight,
    lights: Arc<Mutex<SceneLights>>,
}

impl LightRef {
    fn new(id: usize, light: SceneLight, lights: Arc<Mutex<SceneLights>>) -> Self {
        Self { id, light, lights }
    }

    pub fn get(&self) -> &SceneLight {
        &self.light
    }

    pub fn get_mut(&mut self) -> &mut SceneLight {
        &mut self.light
    }

    pub fn translate_x(&mut self, offset: f32) {
        let translation = Vec3::new(offset, 0.0, 0.0);
        self.translate(translation);
    }

    pub fn translate_y(&mut self, offset: f32) {
        let translation = Vec3::new(0.0, offset, 0.0);
        self.translate(translation);
    }

    pub fn translate_z(&mut self, offset: f32) {
        let translation = Vec3::new(0.0, 0.0, offset);
        self.translate(translation);
    }

    fn translate(&mut self, translation: Vec3) {
        match &mut self.light {
            SceneLight::Spot(l) => {
                let position: Vec3 = Vec3::from(l.position) + translation;
                l.position = position.into();
            }
            _ => {}
        }
    }

    pub fn rotate_x(&mut self, degrees: f32) {
        let rotation = Mat4::from_rotation_x(degrees.to_radians());
        self.rotate(rotation);
    }

    pub fn rotate_y(&mut self, degrees: f32) {
        let rotation = Mat4::from_rotation_y(degrees.to_radians());
        self.rotate(rotation);
    }

    pub fn rotate_z(&mut self, degrees: f32) {
        let rotation = Mat4::from_rotation_z(degrees.to_radians());
        self.rotate(rotation);
    }

    fn rotate(&mut self, rotation: Mat4) {
        match &mut self.light {
            SceneLight::Spot(l) => {
                let direction: Vec3 = l.direction.into();
                let direction = rotation * direction.extend(0.0);
                l.direction = direction.truncate().into();
            }
            SceneLight::Directional(l) => {
                let direction: Vec3 = l.direction.into();
                let direction = rotation * direction.extend(0.0);
                l.direction = direction.truncate().into();
            }
            _ => {}
        }
    }

    pub fn synchronize(&self) -> Result<(), ()> {
        if let Ok(mut lights) = self.lights.try_lock() {
            match &self.light {
                SceneLight::Point(l) => {
                    lights.point_lights[self.id] = l.clone();
                    lights.pl_changed.set(self.id, true);
                }
                SceneLight::Spot(l) => {
                    lights.spot_lights[self.id] = l.clone();
                    lights.sl_changed.set(self.id, true);
                }
                SceneLight::Directional(l) => {
                    lights.directional_lights[self.id] = l.clone();
                    lights.dl_changed.set(self.id, true);
                }
            }

            Ok(())
        } else {
            Err(())
        }
    }
}

pub struct RenderSystem<T: Sized + Renderer> {
    scene: TriangleScene,
    renderer: Arc<Mutex<Box<T>>>,
    pub world: World,
}

pub struct SynchronizeJob<T: Sized + Renderer> {
    renderer: Arc<Mutex<Box<T>>>,
}

unsafe impl<T: Sized + Renderer> Send for SynchronizeJob<T> {}

unsafe impl<T: Sized + Renderer> Sync for SynchronizeJob<T> {}

impl<'a, T: Sized + Renderer> System<'a> for SynchronizeJob<T> {
    type SystemData = ();

    fn run(&mut self, data: Self::SystemData) {
        if let Ok(mut renderer) = self.renderer.lock() {
            renderer.synchronize();
        }
    }
}

pub struct InstanceECS<T: Sized + Renderer> {
    scene: Arc<Mutex<InstancedObjects>>,
    renderer: Arc<Mutex<Box<T>>>,
}

unsafe impl<T: Sized + Renderer> Send for InstanceECS<T> {}

unsafe impl<T: Sized + Renderer> Sync for InstanceECS<T> {}

impl<'a, T: Sized + Renderer> System<'a> for InstanceECS<T> {
    type SystemData = WriteStorage<'a, InstanceRef>;

    fn run(&mut self, mut data: Self::SystemData) {
        let scene = self.scene.lock();
        let renderer = self.renderer.lock();

        if scene.is_err() || renderer.is_err() {
            return;
        }

        let mut scene = scene.unwrap();
        let mut renderer = renderer.unwrap();

        for instance in (&mut data).join() {
            if !instance.changed {
                continue;
            }

            scene.instances[instance.id].set_transform(instance.trs_matrix());
            renderer.set_instance(instance.id, &scene.instances[instance.id]);
            instance.changed = false;
        }
    }
}

impl<T: Sized + Renderer> RenderSystem<T> {
    pub fn new<B: raw_window_handle::HasRawWindowHandle>(
        window: &B,
        width: usize,
        height: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let renderer = T::init(window, width, height)?;
        let mut world = World::new();
        world.register::<InstanceRef>();

        Ok(Self {
            scene: TriangleScene::new(),
            renderer: Arc::new(Mutex::new(renderer)),
            world,
        })
    }

    pub fn resize<B: raw_window_handle::HasRawWindowHandle>(
        &self,
        window: &B,
        width: usize,
        height: usize,
    ) {
        self.renderer.lock().unwrap().resize(window, width, height);
    }

    pub fn render(&self, camera: &Camera, mode: RenderMode) {
        if let Ok(mut renderer) = self.renderer.try_lock() {
            renderer.render(camera, mode);
        }
    }

    pub fn load_mesh<B: AsRef<Path>>(&self, path: B) -> Option<usize> {
        futures::executor::block_on(self.scene.load_mesh(path))
    }

    pub fn add_material<B: Into<[f32; 3]>>(
        &self,
        color: B,
        roughness: f32,
        specular: B,
        transmission: f32,
    ) -> Option<usize> {
        if let Ok(mut materials) = self.scene.materials.lock() {
            Some(materials.add(color, roughness, specular, transmission))
        } else {
            None
        }
    }

    pub fn add_object<B: Into<Mesh>>(&self, object: B) -> Option<usize> {
        self.scene.add_object(object.into())
    }

    pub fn add_instance(&mut self, object: usize) -> Result<Entity, triangle_scene::SceneError> {
        let id = self.scene.add_instance(object, Mat4::identity())?;
        Ok(self
            .world
            .create_entity()
            .with(InstanceRef::new(id))
            .build())
    }

    /// Will return a reference to the point light if the scene is not locked
    pub fn add_point_light<B: Into<[f32; 3]>>(&self, position: B, radiance: B) -> Option<LightRef> {
        if let Ok(mut lights) = self.scene.lights_lock() {
            let position: Vec3 = Vec3::from(position.into());
            let radiance: Vec3 = Vec3::from(radiance.into());

            let light = PointLight::new(position.into(), radiance.into());
            lights.point_lights.push(light.clone());
            lights.pl_changed.push(true);

            let light = LightRef::new(
                lights.point_lights.len() - 1,
                SceneLight::Point(light),
                self.scene.get_lights(),
            );

            Some(light)
        } else {
            None
        }
    }

    /// Will return a reference to the spot light if the scene is not locked
    pub fn add_spot_light<B: Into<[f32; 3]>>(
        &self,
        position: B,
        direction: B,
        radiance: B,
        inner_degrees: f32,
        outer_degrees: f32,
    ) -> Option<LightRef> {
        if let Ok(mut lights) = self.scene.lights_lock() {
            let position = Vec3::from(position.into());
            let direction = Vec3::from(direction.into());
            let radiance = Vec3::from(radiance.into());

            let light = SpotLight::new(
                position.into(),
                direction.into(),
                inner_degrees,
                outer_degrees,
                radiance.into(),
            );

            lights.spot_lights.push(light.clone());
            lights.sl_changed.push(true);

            let light = LightRef::new(
                lights.spot_lights.len() - 1,
                SceneLight::Spot(light),
                self.scene.get_lights(),
            );

            Some(light)
        } else {
            None
        }
    }

    /// Will return a reference to the directional light if the scene is not locked
    pub fn add_directional_light<B: Into<[f32; 3]>>(
        &self,
        direction: B,
        radiance: B,
    ) -> Option<LightRef> {
        if let Ok(mut lights) = self.scene.lights_lock() {
            let light =
                DirectionalLight::new(Vec3::from(direction.into()), Vec3::from(radiance.into()));
            lights.directional_lights.push(light.clone());

            lights.dl_changed.push(true);
            let light = LightRef::new(
                lights.directional_lights.len() - 1,
                SceneLight::Directional(light),
                self.scene.get_lights(),
            );

            Some(light)
        } else {
            None
        }
    }

    pub fn synchronize(&self) {
        if let Ok(mut renderer) = self.renderer.try_lock() {
            let mut changed = false;
            let mut update_lights = false;
            if let Ok(mut objects) = self.scene.objects_lock() {
                if objects.objects_changed.any() {
                    for i in 0..objects.objects.len() {
                        if !objects.objects_changed.get(i).unwrap() {
                            continue;
                        }

                        let mesh = &objects.objects[i];
                        renderer.set_mesh(i, mesh);
                    }
                    objects.objects_changed.set_all(false);
                    changed = true;
                }

                if objects.instances_changed.any() {
                    if let Ok(materials) = self.scene.materials_lock() {
                        let mut found_light = false;
                        let light_flags = materials.light_flags();
                        for i in 0..objects.instances.len() {
                            let object_id = objects.instances[i].get_hit_id();
                            for j in 0..objects.objects[object_id].meshes.len() {
                                if let Some(flag) = light_flags
                                    .get(objects.objects[object_id].meshes[j].mat_id as usize)
                                {
                                    if *flag {
                                        found_light = true;
                                        break;
                                    }
                                }
                            }

                            if found_light {
                                break;
                            }
                        }

                        update_lights = update_lights || found_light;
                    }
                }

                if objects.instances_changed.any() {
                    for i in 0..objects.instances.len() {
                        if !objects.instances_changed.get(i).unwrap() {
                            continue;
                        }

                        let instance = &objects.instances[i];
                        renderer.set_instance(i, instance);
                    }
                    objects.instances_changed.set_all(false);
                    changed = true;
                }
            }

            if let Ok(mut materials) = self.scene.materials_lock() {
                let mut mat_changed = false;
                if materials.textures_changed() {
                    renderer.set_textures(materials.textures_slice());
                    changed = true;
                    mat_changed = true;
                }

                if materials.changed() {
                    let device_materials = materials.into_device_materials();
                    renderer.set_materials(materials.as_slice(), device_materials.as_slice());
                    changed = true;
                    mat_changed = true;
                }

                materials.reset_changed();
                update_lights = update_lights || mat_changed;
            };

            if update_lights {
                self.scene.update_lights();
            }

            if let Ok(mut lights) = self.scene.lights_lock() {
                if lights.pl_changed.any() {
                    renderer.set_point_lights(&lights.pl_changed, lights.point_lights.as_slice());
                    lights.pl_changed.set_all(false);
                    changed = true;
                }

                if lights.sl_changed.any() {
                    renderer.set_spot_lights(&lights.sl_changed, lights.spot_lights.as_slice());
                    lights.sl_changed.set_all(false);
                    changed = true;
                }

                if lights.al_changed.any() {
                    renderer.set_area_lights(&lights.al_changed, lights.area_lights.as_slice());
                    lights.al_changed.set_all(false);
                    changed = true;
                }

                if lights.dl_changed.any() {
                    renderer.set_directional_lights(
                        &lights.dl_changed,
                        lights.directional_lights.as_slice(),
                    );
                    lights.dl_changed.set_all(false);
                    changed = true;
                }
            }
        }

        let mut dispatcher = DispatcherBuilder::new()
            .with(
                InstanceECS {
                    renderer: self.renderer.clone(),
                    scene: self.scene.objects(),
                },
                "update-instances",
                &[],
            )
            .with(
                SynchronizeJob {
                    renderer: self.renderer.clone(),
                },
                "synchronize",
                &["update-instances"],
            )
            .build();

        dispatcher.dispatch(&self.world);
    }

    pub fn get_settings(&self) -> Result<Vec<Setting>, ()> {
        if let Ok(renderer) = self.renderer.try_lock() {
            Ok(renderer.get_settings())
        } else {
            Err(())
        }
    }

    pub fn set_setting(&self, setting: Setting) -> Result<(), ()> {
        if let Ok(mut renderer) = self.renderer.try_lock() {
            renderer.set_setting(setting);
            Ok(())
        } else {
            Err(())
        }
    }

    pub fn get_component<C: Component, F: FnMut(Option<&C>)>(&self, entity: Entity, mut cb: F) {
        let data = self.world.system_data::<ReadStorage<C>>();
        cb(data.get(entity));
    }

    pub fn get_component_mut<C: Component, F: FnMut(Option<&mut C>)>(
        &self,
        entity: Entity,
        mut cb: F,
    ) {
        let mut data = self.world.system_data::<WriteStorage<C>>();
        cb(data.get_mut(entity))
    }
}
