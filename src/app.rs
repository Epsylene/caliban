use winit::window::Window;

pub struct App;

impl App {
    pub fn create(window: &Window) -> Self {
        App
    }

    pub fn render(&mut self, window: &Window) {
        // Render the app here
    }

    pub fn destroy(&mut self) {
        // Destroy the app here
    }
}