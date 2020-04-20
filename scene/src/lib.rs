pub mod material;
pub mod scene;
pub mod triangle_scene;
pub mod objects;
pub mod constants;
pub mod camera;
pub mod lights;

mod utils;

pub use camera::*;
pub use material::*;
pub use scene::*;
pub use triangle_scene::*;
pub use objects::*;
pub use lights::*;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
