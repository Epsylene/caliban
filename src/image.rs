use vulkanalia::{
    prelude::v1_0::*,
    vk::DeviceV1_3,
};
use anyhow::Result;

pub unsafe fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspects: vk::ImageAspectFlags,
    mip_levels: u32,
) -> Result<vk::ImageView> {
    // Images in Vulkan are not accessed as such, but through
    // what are called "image views", which add a level of
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

pub unsafe fn transition_image_layout(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    // Sometimes, the layout of an image has to be changed in
    // order to copy data from a buffer into it (tipically,
    // changing from the initial UNDEFINED to the layout of the
    // pixel data). One of the most common ways to perform
    // layout transitions is using an "image memory barrier".
    // In general, a pipeline barrier is used to synchronize
    // access to resources in the pipeline, like ensuring that
    // a write to a buffer completes before reading from it. An
    // image memory barrier does this, but for an image layout
    // transition. To build the barrier, we need first to
    // define three things:
    //  - Pipeline stages masks (in what stages lie the two
    //    sides of the barrier): we have to set the stages
    //    blocked before the barrier (source stage) and the
    //    ones blocked after (destination stage). Special
    //    values are TOP_OF_PIPE (everything before),
    //    BOTTOM_OF_PIPE (everything after), and ALL_COMMANDS
    //    (all stages);
    //  - Access masks (how the resource is accessed in both
    //    sides of the barrier): the memory is rewritten, so we
    //    need a MEMORY_WRITE flag on both sides, plus
    //    MEMORY_READ on the destination for further
    //    operations.
    let barrier = vk::ImageMemoryBarrier2::builder()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(old_layout)
        .new_layout(new_layout);

    // The aspect mask specifies which types of data are
    // contained in the image (color, depth, stencil, etc),
    // which depends on the new layout.
    let aspect = match new_layout {
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => vk::ImageAspectFlags::DEPTH,
        _ => vk::ImageAspectFlags::COLOR,
    };

    // Finally, the barrier is built with the subresource range
    // and the image to be transitioned.
    let range = subresource_range(aspect);
    let barrier = barrier
        .subresource_range(range)
        .image(image)
        .build();

    // Then, the barrier is inserted into a dependency info
    // struct, which is then passed to the command buffer.
    let barriers = &[barrier];
    let dependency = vk::DependencyInfoKHR::builder()
        .image_memory_barriers(barriers);

    device.cmd_pipeline_barrier2(command_buffer, &dependency);
    
    Ok(())
}

pub fn subresource_range(aspects: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    // A subresource range describes an image's purpose and
    // which parts of the image should be accessed, among other
    // things:
    // - aspect_mask: the part of the image data (the image
    //   aspect) to be accessed (one image could contain both
    //   RGB data and depth info bits, for example);
    // - base_mip_level: the first accessible mipmap level;
    // - level_count: the number of accessible mipmap levels;
    // - base_array_layer: the first accessible array layer
    //   (simultaneous views of the same image, for example for
    //   stereoscopic rendering);
    // - layer_count: the number of accessible array layers.
    vk::ImageSubresourceRange::builder()
        .aspect_mask(aspects)
        .base_mip_level(0)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .base_array_layer(0)
        .layer_count(vk::REMAINING_ARRAY_LAYERS)
        .build()
}