use crate::{
    app::AppData,
    queues::QueueFamilyIndices,
    vertex::INDICES,
};

use vulkanalia::prelude::v1_0::*;
use anyhow::Result;
use log::info;

pub unsafe fn create_command_pool(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
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
    //    and PROTECTED (command buffers which are stored in
    //    "protected memory", preventing unauthorized write or
    //    access);
    //  - Queue family index, which specifies the queue family
    //    corresponding to the type of commands the command
    //    buffers allocated in the pool will record.
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::empty())
        .queue_family_index(indices.graphics);
    
    data.command_pool = device.create_command_pool(&info, None)?;

    info!("Command pool created.");
    Ok(())
}

pub unsafe fn create_command_buffers(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Command buffers are allocated from a command pool, and
    // then recorded with commands. All GPU commands have to go
    // through command buffers, which are then submitted to a
    // queue to be executed.
    //
    // The command buffers allocation takes three parameters:
    //  - The command pool they are allocated in;
    //  - The level of the buffers: this can be either PRIMARY
    //    (command buffers that can be submitted directly to a
    //    Vulkan queue to be executed) or SECONDARY (buffers
    //    that are executed indirectly by being called from
    //    primary command buffers and may not be submitted to
    //    queues). Primary command buffers are the main command
    //    buffers, tied to a single render pass and defining its
    //    structure, as multiple primary command buffers may not
    //    be executed within the same render pass instance.
    //    Secondary command buffers, however, execute within a
    //    specific subpass, which allows threading rendering
    //    operations on a framebuffer;
    //  - The number of command buffers to allocate: when a
    //    command buffer is submitted for execution, it goes
    //    into pending state, which means that it cannot be
    //    reset, and thus that it cannot be re-recorded. This
    //    means that a single command buffer for several
    //    framebuffers would have to wait for the previous frame
    //    to finish before working on the next one. To avoid
    //    this, we will allocate one command buffer per
    //    framebuffer.
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(data.framebuffers.len() as u32);

    data.command_buffers = device.allocate_command_buffers(&allocate_info)?;

    // Command buffers can then be started recording, specifying
    // usage with some parameters:
    //  - flags: either ONE_TIME_SUBMIT (the command buffer will
    //    be rerecorded right after executing it once),
    //    RENDER_PASS_CONTINUE (secondary command buffers that
    //    are entirely within a single render pass) and
    //    SIMULTANEOUS_USE (the command buffer can be
    //    resubmitted while it is in the pending state);
    //  - inheritance info: only used for secondary command
    //    buffer, this specifies which state to inherit from the
    //    calling primary command buffer.
    for (i, &command_buffer) in data.command_buffers.iter().enumerate() {
        let inheritance = vk::CommandBufferInheritanceInfo::builder();
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::empty())
            .inheritance_info(&inheritance);

        device.begin_command_buffer(command_buffer, &info)?;

        // Now that the command buffer is recording, we can
        // start the render pass, by specifying the render area
        // (the swapchain extent, which is the size of the
        // frame)...
        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(data.swapchain_extent);

        // ...the clear color (black)...
        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        // ...and filling the info struct. The render pass can
        // then begin with the corresponding command.
        let clear_values = &[color_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(data.render_pass)
            .framebuffer(data.framebuffers[i])
            .render_area(render_area)
            .clear_values(clear_values);

        // The first parameter (the same for every "cmd_xx"
        // function) is the command buffer we are recording the
        // command to. The second specifies the render pass
        // info, and the third controls how the drawing commands
        // within the render pass will be provided:
        //  - INLINE: the commands are embedded in the primary
        //    command buffer, no secondary command buffer is
        //    used;
        //  - SECONDARY_COMMAND_BUFFERS: the commands will be
        //    executed from secondary command buffers.
        device.cmd_begin_render_pass(command_buffer, &info, vk::SubpassContents::INLINE);
    
        // We can then bind the pipeline, specifying if it is a
        // GRAPHICS or COMPUTE pipeline.
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, data.pipeline);

        // Before finally drawing the triangle, we first have
        // to bind the vertex buffers containing the vertex
        // data for our triangle, as well as the index buffer
        // containing the indices for each vertex in the
        // buffer. The vertex buffer needs to be specified the
        // first vertex input binding to be updated (0), the
        // array of buffers to update (data.vertex_buffer) and
        // the offsets in the buffers (0 here). The index
        // buffer, apart from its data, takes an offset (0 too)
        // and a type size (UINT16 in our case).
        device.cmd_bind_vertex_buffers(command_buffer, 0, &[data.vertex_buffer], &[0]);
        device.cmd_bind_index_buffer(command_buffer, data.index_buffer, 0, vk::IndexType::UINT16);
        
        // Then we can bind the descriptor sets holding the
        // resources passed to the shaders like uniform
        // buffers, specifying the pipeline point they will be
        // used by (GRAPHICS), the pipeline layout, the first
        // set to be bound (0) and the array of sets to bind
        // (in our case the descriptor set attached to the
        // current image). The last parameter is an array of
        // offsets used by dynamic descriptors, which we will
        // not use for now.
        device.cmd_bind_descriptor_sets(
            command_buffer, 
            vk::PipelineBindPoint::GRAPHICS, 
            data.pipeline_layout, 
            0, 
            &[data.descriptor_sets[i]],
            &[]);
        
        // The final draw command takes the length of the index
        // buffer, the number of instances (1 in our case,
        // where we are not doing instanced rendering), the
        // first vertex index in the vertex buffer (0, no
        // offset) and the first instance index (same).
        device.cmd_draw_indexed(command_buffer, INDICES.len() as u32, 1, 0, 0, 0);

        // The render pass can then be ended, and the command
        // buffer can stop recording.
        device.cmd_end_render_pass(command_buffer);
        device.end_command_buffer(command_buffer)?;
    }

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
        .command_pool(data.command_pool)
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
    device.free_command_buffers(data.command_pool, command_buffers);

    Ok(())
}