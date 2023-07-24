use crate::{
    app::AppData,
    queues::QueueFamilyIndices,
};

use vulkanalia::prelude::v1_0::*;
use anyhow::Result;
use log::*;

pub unsafe fn create_framebuffers(
    device: &Device, 
    data: &mut AppData
) -> Result<()> {
    // Each GPU frame can have a number of attachments
    // associated to it, like color, depth, etc. The render pass
    // describes the nature of these attachments, but the object
    // used to actually bind them to an image is the
    // framebuffer. In other words, a framebuffer provides the
    // attachments that a render pass needs while rendering.
    // Attachments can be shared between framebuffers: for
    // example, two framebuffers could have two different color
    // buffer attachments (representing two different swapchain
    // frames) but only one depth buffer (which does not need to
    // be recreated for each frame).
    data.framebuffers = data
        .swapchain_image_views
        .iter()
        .map(|i| {
            // We only have one attachment per framebuffer for
            // now, because we are only rendering to the color
            // attachment. However, since we need to be able to
            // write to each image independently (because we
            // don't know in advance which frame will be
            // presented at a time), we have to create one
            // framebuffer for each image in the swapchain.
            let images = &[*i];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(data.render_pass)
                .attachments(images)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                .layers(1);

            device.create_framebuffer(&create_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    info!("Framebuffers created.");
    Ok(())
}

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
    //    buffers that are rerecorded with new command very
    //    often), RESET_COMMAND_BUFFER (allow command buffers to
    //    be rerecorded individually rather than globally) and
    //    PROTECTED (command buffers which are stored in
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
    //  - The command pool they are allocated on;
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

        // And, finally, draw the triangle. Apart from the
        // command buffer, we have to tell the number of
        // vertices (3), the number of instances (1 in our case,
        // where we are not doing instaced rendering), the first
        // vertex index in the vertex buffer (0, no offset) and
        // the first instance index (idem).
        device.cmd_draw(command_buffer, 3, 1, 0, 0);

        // The render pass can then be ended, and the command
        // buffer can stop recording.
        device.cmd_end_render_pass(command_buffer);
        device.end_command_buffer(command_buffer)?;
    }

    info!("Command buffers created.");
    Ok(())
}