use crate::{
    app::AppData,
    buffers::{create_buffer, find_memory_type}, commands::{begin_single_command_batch, end_single_command_batch},
};

use std::fs::File;
use std::ptr::copy_nonoverlapping as memcpy;

use vulkanalia::prelude::v1_0::*;
use anyhow::{Result, anyhow};
use log::info;

unsafe fn create_texture_image(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // First we open the file, decode it, and retrieve the
    // pixel data as well as some info.
    let image = File::open("res/texture.png")?;

    let decoder = png::Decoder::new(image);
    let mut reader = decoder.read_info()?;

    let mut pixels = vec![0; reader.info().raw_bytes()];
    reader.next_frame(&mut pixels)?;

    let size = reader.info().raw_bytes() as u64;
    let (width, height) = reader.info().size();

    // Then we create a staging buffer in host memory, that
    // will be used to initially hold the pixel data before
    // transfering it to the GPU.
    let (staging_buffer, staging_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // Then, map the memory...
    let memory = device.map_memory(
        staging_memory,
        0,
        size,
        vk::MemoryMapFlags::empty(),
    )?;

    // ...and copy the pixel data into it.
    memcpy(pixels.as_ptr(), memory.cast(), pixels.len());
    device.unmap_memory(staging_memory);

    // Then, the image and its memory are created and bound
    // together with the `create_image()` function. For a
    // texture image, we want in particular a 32-bit SRGBA
    // format, optimally tiled (memory packed), used as the
    // destination of a transfer operation, sampled (to use in
    // shaders), and stored on the GPU.
    let (tex_image, tex_image_memory) = create_image(
        instance,
        device,
        data,
        width,
        height,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.texture_image = tex_image;
    data.texture_image_memory = tex_image_memory;

    // Then, the image is transitioned to a layout that is
    // optimal for the GPU...
    transition_image_layout(
        device, 
        data, 
        data.texture_image, 
        vk::Format::R8G8B8A8_SRGB, 
        vk::ImageLayout::UNDEFINED, 
        vk::ImageLayout::TRANSFER_DST_OPTIMAL
    )?;

    // ...and the pixel data is copied into it.
    copy_buffer_to_image(
        device,
        data,
        staging_buffer,
        data.texture_image,
        width,
        height,
    )?;

    // Finally, we need another transition to make the image
    // ready for sampling.
    transition_image_layout(
        device, 
        data, 
        data.texture_image, 
        vk::Format::R8G8B8A8_SRGB, 
        vk::ImageLayout::TRANSFER_DST_OPTIMAL, 
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    )?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_memory, None);

    info!("Texture image created.");
    Ok(())
}

unsafe fn create_image(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    width: u32,
    height: u32,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    // First, fill the info struct, with:
    // - the image type (2D)
    // - the extent (width, height and depth of the image)
    // - the format (32-bit RGBA SRGB, for example)
    // - the number of mip levels (1)
    // - the number of image layers (for cube maps, environment
    //   maps, animated textures, etc; 1)
    // - the number of samples (for multisampling; 1,
    //   equivalent to no multisampling)
    // - the tiling mode: can be LINEAR (row-major order like
    //   the array of pixels) or OPTIMAL (implementation
    //   defined for optimal GPU access)
    // - the initial layout: for a newly created image like
    //   this one, either UNDEFINED (unknown layout, the image
    //   is treated as containing no valid data, and whatever
    //   it contains will be discarded after the first
    //   transition) or PREINITIALIZED (still not usable by the
    //   GPU, but the transition preserves the texels: this is
    //   inteded for images which need to be written by the
    //   host before staging)
    // - the usage flags: intended usage of the image (as a
    //   transfer source or destination, as storage, as a color
    //   attachement, a depth buffer attachment, etc)
    // - the sharing mode: EXCLUSIVE means that the image is
    //   owned by one queue family at a time, and ownership
    //   must be explicitly transfered before using it in
    //   another queue family.
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .format(format)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::_1)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    // Then we can create the image and retrieve its memory
    // requirements.
    let image = device.create_image(&info, None)?;
    let requirements = device.get_image_memory_requirements(image);
    
    // Creating the memory for the image requires an allocation
    // size (given by the requirements) and a memory type to
    // allocate from (specified both by the image requirements
    // and the given properties).
    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(find_memory_type(
            instance,
            data,
            properties,
            requirements,
        )?);

    // Finally, the memory is allocated and bound to the image.
    let image_memory = device.allocate_memory(&info, None)?;
    device.bind_image_memory(image, image_memory, 0)?;

    Ok((image, image_memory))
}

unsafe fn transition_image_layout(
    device: &Device,
    data: &mut AppData,
    image: vk::Image,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    let command_buffer = begin_single_command_batch(device, data)?;

    // Sometimes, the layout of an image has to changed in
    // order to copy data from a buffer into it (tipically,
    // changing from the initial UNDEFINED to the layout of the
    // pixel data). One of the most common ways to perform
    // layout transitions is using an "image memory barrier".
    // In general, a pipeline barrier is used to synchronize
    // access to resources in the pipeline, like ensuring that
    // a write to a buffer completes before reading from it. An
    // image memory barrier does this, but for an image layout
    // transition. Before building the barrier, however, we
    // need to define the access masks (how the resource is
    // accessed in both sides of the barrier) and the pipeline
    // stages masks (in what stages lie the two sides of the
    // barrier).
    let (
        src_access_mask,
        dst_access_mask,
        src_stage_mask,
        dst_stage_mask,
    ) = match (old_layout, new_layout) {
        // The first transition is from the UNDEFINED layout (a
        // non-specified layout, the one of the pixel data in
        // the staging buffer before copying, that can be
        // safely discarded) to TRANSFER_DST_OPTIMAL (the
        // optimal layout for the destination of a transfer
        // operation):
        // - Access masks: source has no special access flag,
        // destination is a transfer write (write access in a
        // copy operation, like the one we are doing between
        // the staging buffer and the texture image)
        // - Pipeline stage masks: source is TOP_OF_PIPE (the
        // earliest possible one), and destination is TRANSFER
        // (a pseudo-stage where transfer operations happen).
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        // The second transition is to sample the texture for
        // shaders, changing the layout from
        // TRANSFER_DST_OPTIMAL to SHADER_READ_ONLY_OPTIMAL
        // (optimal layout for a read-only access in shaders):
        // - Access masks: source is a transfer write,
        //   destination is a shader read access
        // - Pipeline stage masks: source is TRANSFER,
        //   destination is FRAGMENT_SHADER (we start at the
        //   transfer pseudo-stage and end in the fragment
        //   shader stage).
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        // In all other cases, return an error.
        _ => return Err(anyhow!("Unsupported layout transition!")),
    };
    
    // Apart from the layout, the queue family itself might be
    // changed, so we need to explicitly tell the barrier to
    // ignore these fields. Secondly, a subresource range
    // specifying the part of the image that is affected by the
    // transition is specified. And thirdly, because barriers
    // are primarily used for synchronization purposes, we must
    // specify which types of operations involving the resource
    // mut happen before the barrier, and which must happen
    // after.
    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask);

    // Then, the barrier is submitted using the corresponding
    // command, which takes:
    //  - the command buffer to submit the barrier to;
    //  - the source stage in which the barrier is submitted
    //    (the stage in which the operations that must happen
    //    before the barrier are executed);
    //  - the destination stage in which operations will wait
    //    on the barrier;
    //  - the dependency flag, specifying how execution and
    //    memory dependencies are formed. In Vulkan 1.0, this
    //    is either the empty flag or BY_REGIONS, meaning that
    //    the dependencies between framebuffer-space stages
    //    (the pipeline stages that operate or depend on the
    //    framebuffer, namely the fragment shader, the early
    //    and late fragment tests, and the color attachment
    //    stages) are either framebuffer-global (dependent on
    //    the whole set of stages) or split into multiple
    //    framebuffer-local dependencies;
    //  - the memory barriers, buffer memory barriers and image
    //    memory barriers, which are the sets of barriers
    //    acting between the two stages.
    device.cmd_pipeline_barrier(
        command_buffer, 
        src_stage_mask,
        dst_stage_mask,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier], 
        &[barrier]);

    end_single_command_batch(device, data, command_buffer)?;
    
    Ok(())
}

unsafe fn copy_buffer_to_image(
    device: &Device,
    data: &AppData,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = begin_single_command_batch(device, data)?;

    // Just like with buffer copies, we need to specify which
    // part of the buffer (the "region") is going to be copied
    // to which part of the image (the "subresource").
    let subresource = vk::ImageSubresourceLayers::builder()
    .aspect_mask(vk::ImageAspectFlags::COLOR)
    .mip_level(0)
    .base_array_layer(0)
    .layer_count(1);

    // The row length and image height are used to define the
    // data layout in buffer memory. For example, if row length
    // is 0, then the pixels are tightly packed, and if it is
    // non-zero, then each row of pixels is padded to match the
    // given length.
    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(subresource)
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0})
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    // Then the buffer can be copied to the image, specifying
    // which layout the image is currently using.
    device.cmd_copy_buffer_to_image(
        command_buffer,
        buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );

    end_single_command_batch(device, data, command_buffer)?;
    
    Ok(())
}