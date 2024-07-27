use crate::{
    renderer::RenderData,
    queues::*, 
};

use vulkanalia::prelude::v1_0::*;
use anyhow::Result;
use log::info;

pub unsafe fn create_command_pools(
    instance: &Instance,
    device: &Device,
    data: &mut RenderData,
) -> Result<()> {
    // Commands in Vulkan, like drawing operations and memory
    // transfers, are not executed directly, but recorded in a
    // command buffer and then executed. Command buffers
    // themselves are not allocated directly but within an
    // opaque object called a "command pool", which manages the
    // memory that is used to store the buffers and locks it to
    // a singular thread. Command pool creation takes only two
    // parameters:
    //  - Pool flags, which can be either TRANSIENT (command
    //    buffers that are re-recorded with new commands very
    //    often), RESET_COMMAND_BUFFER (allow command buffers
    //    to be re-recorded individually rather than globally)
    //    or PROTECTED (command buffers which are stored in
    //    "protected memory", preventing unauthorized write or
    //    access);
    //  - Queue family index, which specifies the queue family
    //    corresponding to the type of commands the command
    //    buffers allocated in the pool will record.
    let index = get_graphics_family_index(instance, data.physical_device)?;
    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(index);

    for frame in &mut data.frames {
        let command_pool = device.create_command_pool(&info, None)?;
        frame.command_pool = command_pool;
    }
    
    Ok(())
}

pub unsafe fn create_command_buffers(
    device: &Device,
    data: &mut RenderData,
) -> Result<()> {
    // Command buffers are allocated from a command pool, and
    // then recorded with commands. All GPU commands have to go
    // through command buffers, which are then submitted to a
    // queue to be executed. Each image in the swapchain has
    // its own set of command buffers, which are independent
    // from one another; thus, each image has a command pool to
    // allocate its command buffers from.
    for frame in &mut data.frames {
        // The command buffers allocation takes three
        // parameters:
        //  - The command pool they are allocated from;
        //  - The buffer level: either PRIMARY (command buffers
        //    that can be submitted directly to a Vulkan queue
        //    to be executed) or SECONDARY (buffers that are
        //    executed indirectly by being called from primary
        //    command buffers and may not be submitted to
        //    queues). Primary command buffers are the main
        //    command buffers, tied to a single render pass and
        //    defining its structure, as multiple primary
        //    command buffers may not be executed within the
        //    same render pass instance. Secondary command
        //    buffers, however, execute within a specific
        //    subpass, which allows threading rendering
        //    operations on a framebuffer;
        //  - The number of command buffers to allocate: when a
        //    command buffer is submitted for execution, it
        //    goes into pending state, which means that it
        //    cannot be reset, and thus that it cannot be
        //    re-recorded. This means that a single command
        //    buffer for several framebuffers would have to
        //    wait for the previous frame to finish before
        //    working on the next one; this is the reason why
        //    we allocate one buffer per image.
        //
        // We will start by allocating one primary command
        // buffer for each frame, that will be used to handle
        // the main operations related to the image.
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(frame.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        frame.main_buffer = device.allocate_command_buffers(&allocate_info)?[0];
    }

    info!("Command buffers created.");
    Ok(())
}