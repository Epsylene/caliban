use crate::renderer::Renderer;
use winit::window::Window;
use anyhow::Result;

pub struct App {
    pub renderer: Option<Renderer>,
    pub window: Option<Window>,
    pub minimised: bool,
    pub resized: bool,
}

impl App {
    pub fn new() -> Self {
        App {
            renderer: None,
            window: None,
            minimised: false,
            resized: false,
        }
    }

    /// Initialize the application with the passed window
    /// handle and a new Vulkan renderer.
    pub fn init(&mut self, window: Window) -> Result<()> {
        let renderer = unsafe { Renderer::create(&window)? };
        self.renderer = Some(renderer);
        self.window = Some(window);

        Ok(())
    }

    pub fn destroy(&mut self) {
        if let Some(mut renderer) = self.renderer.take() {
            unsafe { renderer.destroy() };
        }
    }
}