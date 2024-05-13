mod app;
mod queues;
mod devices;
mod swapchain;
mod shaders;
mod pipeline;
mod buffers;
mod vertex;
mod descriptors;
mod texture;
mod commands;
mod image;
mod depth;
mod model;

use vulkanalia::vk::DeviceV1_0;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
    dpi::LogicalSize,
};
use anyhow::Result;
use log::*;

use app::App;

fn main() -> Result<()> {
    std::env::set_var("RUST_LOG", "info");
    pretty_env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("caliban")
        .with_inner_size(LogicalSize::new(1024, 576))
        .build(&event_loop)
        .unwrap();

    let mut app = unsafe { App::create(&window)? };
    let mut destroying = false;
    let mut minimized = false;

    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();
        
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                destroying = true;
                control_flow.set_exit();

                // Render operations are asynchronous, which
                // means that we may call the destroy function
                // before drawing and presentation are
                // completed; to avoid this, we are waiting for
                // the logical device to finish operations
                // before destroying.
                unsafe { app.device.device_wait_idle().unwrap(); }
                unsafe { app.destroy(); }

                info!("Destroyed the app.");
            },
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                .. 
            } => {
                if size.width == 0 || size.height == 0 {
                    minimized = true;
                } else {
                    minimized = false;
                    app.resized = true;
                }
            },
            Event::MainEventsCleared if !destroying && !minimized => {
                // Render the app if the main events are cleared
                // and it is not being destroyed (which is why
                // we use the 'destroying' boolean in the first
                // place) nor it is minimized.
                unsafe { app.render(&window) }.unwrap();
            },
            _ => (),
        }
    })
}