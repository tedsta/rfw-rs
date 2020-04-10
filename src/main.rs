#![allow(dead_code)]

mod camera;
mod cpu_app;
mod gpu_app;
mod utils;

fn main() {
    // let gpu_app = gpu_app::GPUApp::new();
    // fb_template::run_device_app(gpu_app, "GPU App", 1024, 768);
    let cpu_app = cpu_app::CPUApp::new().expect("Could not init App.");
    fb_template::run_host_app(cpu_app, "Rust RT", 512, 512);
}
