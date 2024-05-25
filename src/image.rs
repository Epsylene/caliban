use crate::{
    app::*,
    buffers::*,
    commands::*,
};

use vulkanalia::prelude::v1_0::*;
use anyhow::{Result, anyhow};

pub unsafe fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspects: vk::ImageAspectFlags,
    mip_levels: u32,
) -> Result<vk::ImageView> {
    // Images in Vulkan are not accessed as such, but through
    // what are called "image views", which add level of
    // indirection to the image specifying how it should be
    // accessed. The first element of the view to define is how
    // the image colors are mapped to the image view colors. We
    // don't want to swizzle (map to a different value) the
    // color components here, so we just go for the identity.
    let component_mapping = vk::ComponentMapping::builder()
        .r(vk::ComponentSwizzle::IDENTITY)
        .g(vk::ComponentSwizzle::IDENTITY)
        .b(vk::ComponentSwizzle::IDENTITY)
        .a(vk::ComponentSwizzle::IDENTITY)
        .build();

    // The next thing to consider is the subresource range,
    // which describes the image's purpose and which parts of
    // the image should be accessed, among other things:
    // - aspect_mask: the part of the image data (the image
    //   aspect) to be accessed (one image could contain both
    //   RGB data and depth info bits, for example);
    // - base_mip_level: the first accessible mipmap level;
    // - level_count: the number of accessible mipmap levels;
    // - base_array_layer: the first accessible array layer
    //   (simultaneous views of the same image, for example for
    //   stereoscopic rendering);
    // - layer_count: the number of accessible array layers.
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspects)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1)
        .build();

    // Then we can build the info struct, containing the image
    // itself, the view type of the image (1D, 2D, or 3D
    // texture, cube map), its format, the mapping of the color
    // components between the image and the view, and the
    // subresource range of the image view.
    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(format)
        .components(component_mapping)
        .subresource_range(subresource_range);

    Ok(device.create_image_view(&info, None)?)
}

pub unsafe fn create_image(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    width: u32,
    height: u32,
    mip_levels: u32,
    samples: vk::SampleCountFlags,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    // First, fill the info struct, with:
    // - the image type (2D)
    // - the extent (width, height and depth of the image)
    // - the format (32-bit RGBA SRGB, for example)
    // - the number of mip levels (as much as we calculated the
    //   texture image to have)
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
        .mip_levels(mip_levels)
        .array_layers(1)
        .samples(samples)
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

pub unsafe fn create_color_objects(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // MSAA, by its very definition, needs to store multiple
    // samples per pixel to perform antialiasing. Our
    // framebuffer is not equipped to handle multisampled
    // images, however, so we need to create a separate buffer
    // to write the MSAA operation to, which we call the "color
    // image" (since it holds color data). This image is marked
    // as COLOR_ATTACHMENT (used as color or resolve
    // attachment) and TRANSIENT_ATTACHMENT (optimized for
    // short-lived data, like the MSAA buffer).
    let (color_image, color_image_memory) = create_image(
        instance, 
        device, 
        data, 
        data.swapchain_extent.width, 
        data.swapchain_extent.height, 
        1,
        data.msaa_samples,
        data.swapchain_format, 
        vk::ImageTiling::OPTIMAL, 
        vk::ImageUsageFlags::COLOR_ATTACHMENT 
            | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.color_image = color_image;
    data.color_image_memory = color_image_memory;
    
    // Then we can create the image view for swapchain usage.
    data.color_image_view = create_image_view(
        device, 
        data.color_image, 
        data.swapchain_format, 
        vk::ImageAspectFlags::COLOR,
        1
    )?;
    
    Ok(())
}

pub unsafe fn transition_image_layout(
    device: &Device,
    data: &mut AppData,
    image: vk::Image,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    mip_levels: u32,
) -> Result<()> {
    let command_buffer = begin_single_command_batch(device, data)?;

    // Sometimes, the layout of an image has to be changed in
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
        // In the case of a depth attachment, the layout
        // transition is from UNDEFINED to
        // DEPTH_STENCIL_ATTACHMENT, with:
        // - Access masks: the result of the transition is a
        //   read/write operation on the depth/stencil
        //   attachment
        // - Pipeline stage masks: the depth buffer is read
        //   from to perform depth tests to see if a fragment
        //   is visible (EARLY_FRAGMENT_TESTS stage), and then
        //   is written when a new fragment is drawn
        //   (LATE_FRAGMENT_TESTS). Thus, the destination of
        //   the barrier is the earlier stage.
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
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
    // must happen before the barrier, and which must happen
    // after.
    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: {
                // For the aspect mask, if we are dealing with
                // a depth attachment, we want to match the
                // given image format depending on whether it
                // contains a stencil component or not.
                if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
                    match format {
                        vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT 
                            => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                        _ => vk::ImageAspectFlags::DEPTH
                    }
                // For color attachments, we just want the
                // color aspect flag.
                } else {
                    vk::ImageAspectFlags::COLOR
                }
            },
            base_mip_level: 0,
            level_count: mip_levels,
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

pub unsafe fn copy_buffer_to_image(
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

pub unsafe fn get_supported_format(
    instance: &Instance,
    data: &AppData,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    // For each swapchain image, texture, depth component, etc,
    // there are several possible formats to choose from. We
    // will just search for a format to use that is supported
    // by the device and has the desired features.
    candidates
        .iter()
        .cloned()
        .find(|&f| {
            // We first query the properties of the given
            // format which are present in the physical device.
            let properties = instance.get_physical_device_format_properties(
                data.physical_device, 
                f
            );

            // There are 3 of these properties: linear tiling,
            // optimal tiling, and buffer features (meaning,
            // "is the format supported for this device with
            // linear tiling/optimal tiling/buffers?"). We are
            // interested in the first two, so we check for
            // each if the given format property contains the
            // desired features.
            match tiling {
                vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
                _ => false
            }
        })
        .ok_or_else(|| anyhow!("Failed to find supported format"))
}

pub unsafe fn get_max_msaa_samples(
    instance: &Instance,
    data: &AppData,
) -> vk::SampleCountFlags {
    // The maximum number of samples supported by the device is
    // queried from the physical device properties. There are
    // both color and depth sample counts, so we will take the
    // highest common value.
    let properties = instance.get_physical_device_properties(data.physical_device);
    let counts = properties.limits.framebuffer_color_sample_counts
        & properties.limits.framebuffer_depth_sample_counts;

    // The number of samples is a bitmask, so we can just
    // iterate over the possible values and return the first
    // one that is supported. If none are, we just return 1
    // (one sample per pixel, the same as no multisampling).
    [
        vk::SampleCountFlags::_64,
        vk::SampleCountFlags::_32,
        vk::SampleCountFlags::_16,
        vk::SampleCountFlags::_8,
        vk::SampleCountFlags::_4,
        vk::SampleCountFlags::_2,
    ]
    .iter()
    .cloned()
    .find(|&c| counts.contains(c))
    .unwrap_or(vk::SampleCountFlags::_1)
}