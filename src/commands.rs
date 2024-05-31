use crate::{
    app::AppData, 
    queues::QueueFamilyIndices
};

use vulkanalia::prelude::v1_0::*;
use glam::{Mat4, Vec3};
use anyhow::Result;
use log::info;

pub unsafe fn create_command_pools(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Rather than having a single command pool for all frames,
    // we will have one command pool per image in the
    // swapchain. This is because command buffers are submitted
    // to queues, and we want to avoid having to wait for the
    // queue to finish executing the commands before we can
    // record new ones. By having one command pool per image,
    // we can record commands for all images in parallel, and
    // submit them to the queue as soon as they are ready. We
    // do want however one global command pool for commands
    // that are not tied to a specific image, like allocating
    // command buffers during initialization.
    data.global_command_pool = create_command_pool(instance, device, data)?;
    
    // For each image in the swapchain, we create one command
    // pool to allocate command buffers from.
    let num_images = data.swapchain_images.len();
    for _ in 0..num_images {
        let command_pool = create_command_pool(instance, device, data)?;
        data.command_pools.push(command_pool);
    }

    info!("Command pools created.");
    Ok(())
}

pub unsafe fn create_command_pool(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<vk::CommandPool> {
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
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(indices.graphics);

    let command_pool = device.create_command_pool(&info, None)?;
    
    Ok(command_pool)
}

pub unsafe fn create_command_buffers(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Command buffers are allocated from a command pool, and
    // then recorded with commands. All GPU commands have to go
    // through command buffers, which are then submitted to a
    // queue to be executed. Each image in the swapchain has
    // its own set of command buffers, which are independent
    // from one another; thus, each image has a command pool to
    // allocate its command buffers from.
    for &command_pool in &data.command_pools {
        // The command buffers allocation takes three parameters:
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
        // buffer for each image in the swapchain, that will be
        // used to handle the main operations related to the
        // image.
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = device.allocate_command_buffers(&allocate_info)?[0];
        data.primary_command_buffers.push(command_buffer);
    }

    // Then, for each image we will have a set of secondary
    // command buffers. Like primary command buffers, secondary
    // command buffers can be allocated, recorded and freed
    // independently; however, primary command buffers cannot
    // be executed within the same render pass. Secondary
    // command buffers are not tied to a specific render pass
    // instance, so they can be executed in parallel. We will
    // have an array of as many lists of secondary command
    // buffers as there are images in the swapchain, so that
    // each can manage its own set of secondary command
    // buffers.
    data.secondary_command_buffers = vec![vec![]; data.swapchain_images.len()];

    info!("Command buffers created.");
    Ok(())
}

pub unsafe fn begin_single_command_batch(
    device: &Device,
    data: &AppData,
) -> Result<vk::CommandBuffer> {
    // Before doing anything, we want to allocate one (1)
    // primary command buffer in our pool.
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.global_command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

    // Then we can begin the command buffer, and specify that
    // it is to be used only once.
    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(command_buffer, &begin_info)?;

    Ok(command_buffer)
}

pub unsafe fn end_single_command_batch(
    device: &Device,
    data: &AppData,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    // After recording the commands, we can end the command
    // buffer and submit it to the queue.
    device.end_command_buffer(command_buffer)?;

    // Submitting the command buffer to the queue executes it,
    // and then makes it wait for the queue to complete
    // operations before continuing the program.
    let command_buffers = &[command_buffer];
    let submit_info = vk::SubmitInfo::builder()
        .command_buffers(command_buffers);

    // Note that using a fence instead of queue_wait_idle()
    // would allow to schedule multiple transfers at once and
    // wait for all of them to complete, instead of executing
    // one at a time.
    device.queue_submit(data.graphics_queue, &[submit_info], vk::Fence::null())?;
    device.queue_wait_idle(data.graphics_queue)?;

    // We can then free the command buffer, as it is not
    // needed anymore.
    device.free_command_buffers(data.global_command_pool, command_buffers);

    Ok(())
}