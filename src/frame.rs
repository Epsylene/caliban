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
#[derive(Default)]
pub struct FrameData {
    pub command_pool: vk::CommandPool,
    pub main_buffer: vk::CommandBuffer,
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
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