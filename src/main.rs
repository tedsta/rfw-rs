#![allow(dead_code)]

mod cpu_app;
mod gpu_app;
mod utils;

enum AppType {
    CPURayTracer,
    GPU,
}

use clap::{Arg, App};

fn main() {
    let matches = App::new("rust-rfw")
        .version("0.1.0")
        .author("Mèir Noordermeer <meirnoordermeer@me.com>")
        .arg(Arg::with_name("renderer")
            .short("r")
            .long("renderer")
            .takes_value(true).help("renderer to use"))
        .get_matches();

    let app = matches.value_of("renderer").unwrap_or("");
    let app = if app.starts_with("cpu") {
        AppType::CPURayTracer
    } else {
        AppType::GPU
    };

    let width = 1024;
    let height = 768;

    match app {
        AppType::CPURayTracer => {
            let cpu_app = cpu_app::CPUApp::new().expect("Could not init App.");
            fb_template::run_host_app(cpu_app, "Rust RT", width, height);
        }
        AppType::GPU => {
            let gpu_app = gpu_app::GPUApp::new();
            fb_template::run_device_app(gpu_app, "GPU App", width, height);
        }
    };
}
