use crate::{
    app::AppData,
    buffers::create_buffer,
    commands::*,
    image::*,
};

use std::fs::File;
use std::ptr::copy_nonoverlapping as memcpy;
use std::cmp::max;

use vulkanalia::prelude::v1_0::*;
use anyhow::{anyhow, Result};
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

    // The mipmap texture is composed of the texture repeated
    // multiple times, first at full resolution, then at half,
    // quarter and so on until the last level which has a size
    // of one pixel. The first and subsequent levels are placed
    // to the side or under the original texture; thus, the
    // number of mipmap levels (that is, the number of
    // subtextures) can be calculated as the floor of the log2
    // of the longest side of the texture (the number of times
    // we can divide by 2 that dimension), plus one (for the
    // original image).
    data.mip_levels = (max(width, height) as f32).log2().floor() as u32 + 1;

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
    // format, optimally tiled (memory packed), sampled (to use
    // in shaders), used as both the source and destination of
    // a transfer operation (because of the blit operation to
    // generate the mipmaps), and stored on the GPU.
    let (tex_image, tex_image_memory) = create_image(
        instance,
        device,
        data,
        width,
        height,
        data.mip_levels,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST,
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
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        data.mip_levels,
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

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_memory, None);
    
    // Lastly, the texture mipmaps can be generated.
    generate_mipmaps(
        instance,
        device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        width,
        height,
        data.mip_levels,
    )?;
    
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
        data.mip_levels,
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
        .max_lod(data.mip_levels as f32);

    data.texture_sampler = device.create_sampler(&info, None)?;

    info!("Texture sampler created.");
    Ok(())
}

unsafe fn generate_mipmaps(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<()> {
    let command_buffer = begin_single_command_batch(device, data)?;

    // Before anything, check that the device supports linear
    // filtering, which will be used later for interpolation
    // between mip levels.
    if !instance
        .get_physical_device_format_properties(data.physical_device, format)
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR) 
    {
        return Err(anyhow::anyhow!("Texture image format does not support linear blitting."));
    }

    // Mipmaps are generated from a base image, which we have
    // to define as our subresource, with its range: colored,
    // single-layered, and with a single mip level.
    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    // Blit operations are rendering functions that copy and
    // transform the image unto itself. Bit blits combine at
    // least two bitmaps, the "source" (or "foreground") and
    // the "destination" (or "background"), with a boolean
    // function to produce the final result; scaling, for
    // example, is done by interpolating pixels of the source
    // map in a destination map of a different size. Mipmaps
    // are generated by blitting the image to a smaller image,
    // then blitting the smaller image to an even smaller
    // image, and so on. Thus, we will want to reuse the
    // barrier for layout transitions between mip levels.
    let mut barrier = vk::ImageMemoryBarrier::builder()
        .image(image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(subresource);

    let mut mip_width = width as i32;
    let mut mip_height = height as i32;

    for i in 1..mip_levels {
        // We start by transitioning level i-1 (the base image,
        // then level 1, etc) to a TRANSFER_SRC layout and
        // TRANSFER_READ access mask, that is, a layout optimal
        // for reading the image data (because that is what the
        // blit operation will first do).
        barrier.subresource_range.base_mip_level = i - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

        // This will insert a memory dependency at the barrier,
        // so that the transition waits for level i-1 to be
        // filled, either from the previous blit command or
        // from the copy operation, before continuing.
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        // Then, we can specify the regions that will be used
        // for the blit operation. The source and destination
        // regions have the same layers, except that one is at
        // mip level i-1 and the other at i.
        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i - 1)
            .base_array_layer(0)
            .layer_count(1);

        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i)
            .base_array_layer(0)
            .layer_count(1);

        // Then, the blit operation can be defined.
        let blit = vk::ImageBlit::builder()
            .src_subresource(src_subresource)
            // The source region is the entire image at level
            // i-1, given by the width and height of the mip
            // level.
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width,
                    y: mip_height,
                    z: 1,
                },
            ])
            // The destination region is half of that region,
            // down-clamped to 1 (since the image cannot be
            // less than 1 pixel wide or tall).
            .dst_subresource(dst_subresource)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: max(1, mip_width / 2),
                    y: max(1, mip_height / 2),
                    z: 1,
                },
            ]);

        // The blit operation is then executed. The same image
        // is used as the source and destination, because we're
        // blitting between different levels of the same
        // texture. The filter is set to LINEAR, which means
        // that the pixels will be interpolated linearly
        // between mip levels.
        device.cmd_blit_image(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit],
            vk::Filter::LINEAR,
        );

        // After the blit operation, the image is transitioned
        // to a SHADER_READ layout, which is optimal for
        // sampling the image in shaders.
        barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        // The barrier makes the transition wait on the blit
        // command to finish before continuing.
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        // Finally, the mip level dimensions are halved for the
        // next iteration.
        mip_width = max(1, mip_width / 2);
        mip_height = max(1, mip_height / 2);

        // The loop continues until the mip level is 1x1.
    }

    // There is one more transition to handle, for the last mip
    // level: it is never blitted (halved) to another level,
    // since it is already minimal, but we still need to change
    // its layout from TRANSFER_DST to SHADER_READ_ONLY.
    barrier.subresource_range.base_mip_level = mip_levels - 1;
    barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_command_batch(device, data, command_buffer)?;

    info!("Mipmaps generated.");
    Ok(())
}