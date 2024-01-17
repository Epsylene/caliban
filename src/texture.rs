use crate::{
    app::AppData,
    buffers::{create_buffer, find_memory_type},
};

use std::fs::File;
use std::ptr::copy_nonoverlapping as memcpy;

use vulkanalia::prelude::v1_0::*;
use anyhow::Result;
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

    // First, we create a staging buffer in host memory, that
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
    // format, optimally tiled (memory packing), used as the
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