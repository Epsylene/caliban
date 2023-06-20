mod app;

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
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();
        
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                destroying = true;
                info!("Destroying the app.");
                control_flow.set_exit();
                unsafe { app.destroy(); }
            },
            Event::MainEventsCleared if !destroying => {
                unsafe { app.render(&window) }.unwrap();
            },
            _ => (),
        }
    })
}