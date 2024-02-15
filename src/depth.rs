use crate::{
    app::AppData, 
    image::*
};

use vulkanalia::prelude::v1_0::*;
use anyhow::Result;
use log::info;

pub unsafe fn create_depth_objects(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Depth objects are the depth attachment of an image and
    // its memory. The depth attachment contains information on
    // whether a fragment is occluded by other fragments, which
    // is necessary to determine the final color of a pixel in
    // scenes with multiple layers of geometry. We first want
    // to get the format of the depth attachment that is
    // available for the current device.
    let format = get_depth_format(instance, data)?;

    // Then, we can create the depth image and its memory. From
    // the swapchain point of view, this is just another image,
    // with the same extent as the color attachments, optimal
    // tiling and device local memory, but presented as a depth
    // and stencil attachment (the stencil component stores the
    // results of stencil tests, which will be useful later).
    let (depth_image, depth_image_memory) = create_image(
        instance, 
        device, 
        data, 
        data.swapchain_extent.width, 
        data.swapchain_extent.height, 
        format, 
        vk::ImageTiling::OPTIMAL, 
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, 
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.depth_image = depth_image;
    data.depth_image_memory = depth_image_memory;
    
    // Then, as with other images, we need to create an image
    // view to access the depth attachment from the shader.
    data.depth_image_view = create_image_view(
        device, 
        data.depth_image, 
        format,
        vk::ImageAspectFlags::DEPTH,
    )?;

    transition_image_layout(
        device, 
        data, 
        data.depth_image, 
        format, 
        vk::ImageLayout::UNDEFINED, 
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    )?;

    info!("Depth objects created.");
    Ok(())
}

pub unsafe fn get_depth_format(
    instance: &Instance,
    data: &AppData,
) -> Result<vk::Format> {
    // Depth formats are characterized by their depth
    // (tipically 24- or 32-bits), their data type (SFLOAT for
    // signed floats, UNORM for unsigned normalized floats) and
    // the presence of a stencil component (S8_UINT for 8-bit
    // unsigned integer).
    let depth_formats = &[
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];

    // Then, we can use the helper function to get the first
    // supported format with optimal tiling and a depth/stencil
    // attachment.
    get_supported_format(
        instance, 
        data, 
        depth_formats, 
        vk::ImageTiling::OPTIMAL, 
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}