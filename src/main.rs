mod app;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    dpi::LogicalSize,
};

use app::App;

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("caliban")
        .with_inner_size(LogicalSize::new(1024, 576))
        .build(&event_loop)
        .unwrap();

    let mut app = App::create(&window);
    let mut destroying = false;
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();
        
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                destroying = true;
                control_flow.set_exit();
                app.destroy();
            },
            Event::MainEventsCleared if !destroying => {
                app.render(&window);
            },
            _ => (),
        }
    })
}