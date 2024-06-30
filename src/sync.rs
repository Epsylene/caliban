use crate::app::AppData;

use vulkanalia::prelude::v1_0::*;
use anyhow::Result;
use log::info;

pub unsafe fn create_sync_objects(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Rendering operations, such as acquiring images,
    // presenting images or running a command buffer are
    // executed asynchronously. This means that the order of
    // execution is undefined, which poses a problem because
    // each operation depends on the completion of the previous
    // one. To solve this, Vulkan provides two ways of
    // synchronizing swapchain events: fences and semaphores.
    // Semaphores are simply signal identifiers that indicate
    // when a batch of commands has been processed. Fences are
    // similar to semaphores, but they have accessible state
    // and can be waited for from the program code; thus, they
    // can insert a dependency between a queue and the host,
    // which means that they are used for CPU-GPU
    // synchronization, while semaphores handle GPU-GPU
    // synchronization. We have to take care of setting the
    // SIGNALED flag when creating the fences, because they are
    // in the unsignaled state by default, which will freeze
    // the program when the render function waits for the
    // fences to be signaled the first time.
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder()
        .flags(vk::FenceCreateFlags::SIGNALED);

    for frame in &mut data.frames {
        // In our case, we will need one semaphore to signal
        // that an image has been acquired and is ready for
        // rendering, and one to signal that rendering has
        // finished and presentation can happen.
        frame.image_available_semaphore = device.create_semaphore(&semaphore_info, None)?;
        frame.render_finished_semaphore = device.create_semaphore(&semaphore_info, None)?;
        
        // Furthermore, we need to create a fence for each
        // frame to syncg the CPU with the GPU: if the CPU is
        // submitting work faster than the GPU can process it,
        // semaphores and command buffers will be used for
        // multiple frames at the same time. Creating a fence
        // for each frame in the swapchain allows us to wait
        // for objects to finish executing while having
        // multiple frames "in-flight" (worked on
        // asynchronously).
        frame.in_flight_fence = device.create_fence(&fence_info, None)?;
    }
   
    info!("Sync objects created.");
    Ok(())
}

pub unsafe fn destroy_sync_objects(
    device: &Device,
    data: &mut AppData,
) {
    for frame in &mut data.frames {
        device.destroy_semaphore(frame.image_available_semaphore, None);
        device.destroy_semaphore(frame.render_finished_semaphore, None);
        device.destroy_fence(frame.in_flight_fence, None);
    }

    info!("Sync objects destroyed.");
}

pub unsafe fn semaphore_submit(
    stage_mask: vk::PipelineStageFlags2,
    semaphore: vk::Semaphore,
) -> vk::SemaphoreSubmitInfo {
    // A semaphore submit operation requires a semaphore, a
    // mask of pipeline stages which limit the synchronization
    // scope of the semaphore, the index of the device
    // executing the operation, and a value to either signal or
    // wait on.
    vk::SemaphoreSubmitInfo::builder()
        .semaphore(semaphore)
        .stage_mask(stage_mask)
        .device_index(0)
        .value(1)
        .build()
}