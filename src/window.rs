use crate::app::App;
use vulkanalia::prelude::v1_0::DeviceV1_0;
use winit::{
    application::ApplicationHandler, 
    dpi::LogicalSize, 
    event::WindowEvent, 
    event_loop::ActiveEventLoop,
    window::Window
};

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attr = Window::default_attributes()
            .with_title("caliban")
            .with_inner_size(LogicalSize::new(1024, 576));

        let window = event_loop.create_window(window_attr).unwrap();
        if self.window.is_none() {
            self.init(window).unwrap();
        }
    }

    fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _: winit::window::WindowId,
            event: WindowEvent,
        ) {
        match event {
            WindowEvent::CloseRequested => {
                // Render operations are asynchronous, which
                // means that we may call the destroy function
                // before drawing and presentation are
                // completed; to avoid this, we are waiting for
                // the logical device to finish operations
                // before destroying.
                if let Some(ref device) = self.renderer {
                    unsafe { device.device.device_wait_idle().unwrap(); }
                    self.destroy();
                }

                // Close the window
                event_loop.exit();
            },
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    self.minimised = true;
                } else {
                    self.minimised = false;
                    self.resized = true;
                }
            },
            WindowEvent::RedrawRequested => {
                unsafe { self.renderer.as_mut().unwrap().render().unwrap() };
            },
            _ => (),
        }
    }
}