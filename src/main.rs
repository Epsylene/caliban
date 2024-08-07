mod renderer;
mod devices;
mod queues;
mod swapchain;
mod image;
mod buffers;
mod commands;
mod frame;
mod sync;
mod window;
mod app;
mod allocator;

use winit::event_loop::{EventLoop, ControlFlow};
use app::App;
use anyhow::Result;

fn main() -> Result<()> {
    std::env::set_var("RUST_LOG", "info");
    pretty_env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}