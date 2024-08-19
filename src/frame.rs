use vulkanalia::prelude::v1_0::*;

// Data relative to a single render frame:
//  - Command pool: pool where main buffer is allocated
//  - Main buffer: handle frame commands
//  - Swapchain semaphore: wait from the CPU on a swapchain
//    image request
//  - Render semaphore: wait from the CPU for drawing to finish
//    to present image
//  - In-flight fence: wait on the GPU for the draw commands to
//    complete

/// Data for a single render frame.
#[derive(Default)]
pub struct FrameData {
    /// Command pool where the main buffer is allocated.
    pub command_pool: vk::CommandPool,
    /// Main buffer to handle frame commands.
    pub main_buffer: vk::CommandBuffer,
    /// Semaphore to signal to the host that the image has been
    /// acquired and is ready for rendering.
    pub image_available_semaphore: vk::Semaphore,
    /// Semaphore to signal to the host that rendering has
    /// finished and presentation can happen.
    pub render_finished_semaphore: vk::Semaphore,
    /// Fence to wait for the draw commands on the device to
    /// complete.
    pub in_flight_fence: vk::Fence,
}

impl FrameData {
    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_command_pool(self.command_pool, None);
        device.destroy_semaphore(self.image_available_semaphore, None);
        device.destroy_semaphore(self.render_finished_semaphore, None);
        device.destroy_fence(self.in_flight_fence, None);
    }
}