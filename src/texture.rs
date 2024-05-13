use crate::{
    app::AppData,
    buffers::create_buffer,
    image::*,
};

use std::fs::File;
use std::ptr::copy_nonoverlapping as memcpy;

use vulkanalia::prelude::v1_0::*;
use anyhow::Result;
use log::info;

pub unsafe fn create_texture_image(
    path: &str,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // First we open the file, decode it, and retrieve the
    // pixel data as well as some info.
    let image = File::open(path)?;

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

pub unsafe fn create_texture_image_view(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // The texture image view is created from the image, making
    // sure that the format is 32-bit sRGB.
    data.texture_image_view = create_image_view(
        device,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
    )?;

    info!("Texture image view created.");
    Ok(())
}

pub unsafe fn create_texture_sampler(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Texture sampling is the process of reading textures
    // through the GPU. Instead of reading the image texel by
    // texel, samplers are used to filter and transform the
    // texture data to produce a final color. Creating a
    // sampler requires setting up a few parameters:
    // - Magnification filter: when a single texel affects many
    //   fragments (oversampling, think pixelated images), a
    //   magnification filter is used to upsample the texture;
    //   in this case, we set the filtering to LINEAR, which
    //   combines 4 weighted texel values to produce the final
    //   color.
    // - Minification filter: when many texels affect a single
    //   fragment (undersampling, which happens when sampling
    //   high frequency patterns like checkerboard textures),
    //   the texture has to be downsampled.
    // - Adress mode: for each texel coordinate, adressing
    //   (that is, what to do when the coordinate is outside
    //   the texture range) can be set to REPEAT (wrap around,
    //   creating a tiled pattern), MIRRORED_REPEAT (same as
    //   repeat, but mirrors the texture), CLAMP_TO_EDGE (take
    //   the color of the edge closest to the coordinate),
    //   MIRRORED_CLAMP_TO_EDGE (same but using the opposite
    //   edge) and CLAMP_TO_BORDER (take a user-defined color).
    // - Anisotropy: when the texture is viewed at a steep
    //   angle, the texels are projected to a larger area,
    //   creating a blurry effect; anisotropic filtering
    //   reduces this effect by creating a mipmap of the
    //   texture linearly deformed in each direction. The
    //   maximum anisotropy is set to use 16 samples, which is
    //   the maximum value in graphics hardware today since
    //   differences are negligible beyond this point.
    // - Border color: when the address mode is set to
    //   CLAMP_TO_BORDER, this is the color used to fill the
    //   space; it is either black, white, or transparent.
    // - Unnormalized coordinates: when set to true, the texel
    //   coordinates range from [0,width) and [0,height)
    //   instead of [0,1).
    // - Compare enable/op: whether to enable a comparison
    //   function, with which the texels will first be compared
    //   to a value before being sampled. Here we set the
    //   compare operation to ALWAYS (always return true).
    // - Mipmap mode: how to sample the mipmap levels, either
    //   NEAREST (take the nearest mipmap level) or LINEAR
    //   (linearly interpolate between the two nearest levels).
    // - Mip LOD bias: a bias to add to the LOD level, which is
    //   the number determining the mipmap level (or
    //   combination of levels) being sampled.
    // - Min/max LOD: the range of LOD levels to sample from.
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(0.0);

    data.texture_sampler = device.create_sampler(&info, None)?;

    info!("Texture sampler created.");
    Ok(())
}