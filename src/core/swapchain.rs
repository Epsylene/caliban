use crate::{
    renderer::RenderData,
    core::{queues::*, image::*},
};

use vk::KhrSwapchainExtension;
use vulkanalia::{
    prelude::v1_0::*,
    vk::KhrSurfaceExtension,
};

use log::*;
use anyhow::Result;
use winit::window::Window;

pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub fn get_swapchain_support(
    instance: &Instance,
    data: &RenderData,
    physical_device: vk::PhysicalDevice,
) -> Result<SwapchainSupport> {
    // There is no concept of a "default framebuffer" in Vulkan
    // as there is in OpenGL, so it requires an infrastructure
    // that will own the buffers we will render to before we
    // visualize them on the screen. This is the swapchain,
    // essentially a queue of images that are waiting to be
    // presented to the screen. Not all graphics cards are
    // capable of presenting images directly to a screen (for
    // example because they are designed for servers and don't
    // have any display outputs), so swapchain support and
    // compatibility with our window surface have to be queried
    // beforehand.
    Ok(SwapchainSupport {
        capabilities: unsafe { 
            instance.get_physical_device_surface_capabilities_khr(
                physical_device,
                data.surface,
            )?
        },
        formats: unsafe {
            instance.get_physical_device_surface_formats_khr(
                physical_device,
                data.surface,
            )?
        },
        present_modes: unsafe {
            instance.get_physical_device_surface_present_modes_khr(
                physical_device,
                data.surface,
            )?
        },
    })
}

fn get_swapchain_surface_format(
    formats: &[vk::SurfaceFormatKHR],
) -> vk::SurfaceFormatKHR {
    // The first setting to determine is the surface format,
    // which itself consists of two fields: 'format', which
    // specifies the color channels and types, and
    // 'color_space' which indicates the supported color space.
    // In our case, we will want a B8G8R8A8_SRGB format (B, G,
    // R and alpha channels of 8 bits each in sRGB color space,
    // which makes for 32 bits of color per pixel, the most
    // common bit depth) and a sRGB color space (standard
    // non-linear RGB space, made to match more closely the way
    // the human eye perceives color). If this surface format
    // is not available, we will just default on the first one
    // available.
    formats
        .iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
            && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .cloned()
        .unwrap_or(formats[0])
}

fn get_swapchain_present_mode(
    present_modes: &[vk::PresentModeKHR],
) -> vk::PresentModeKHR {
    // The second property of the swapchain to determine is the
    // presentation mode, which is the way images are sent from
    // the render queue to the screen. There are four possible
    // modes available in Vulkan:
    // - IMMEDIATE: images are submitted right away, which may
    //   result in tearing (since the graphics and display
    //   devices refresh rates may not match)
    // - FIFO: images are queued and presented after each
    //   vertical blanking interval (VBI), when the display is
    //   refreshed. This prevents tearing, and is most similar
    //   to vertical sync (VSync) in OpenGL.
    // - FIFO_RELAXED: like FIFO, but if the application is late
    //   and the queue was empty at the last vertical blank, the
    //   next image is immediately presented to avoid a frame
    //   lag, at the risk of visible tearing.
    // - MAILBOX: like FIFO, but if the queue is full, instead
    //   of blocking the application, the queued images are
    //   simply replaced with newer ones. This is equivalent to
    //   what is commonly known as "triple buffering", which
    //   results in fewer latency with no tearing, but also a
    //   higher CPU and GPU usage.
    present_modes
        .iter()
        .cloned()
        .find(|&m| m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn get_swapchain_extent(
    window: &Window,
    capabilities: vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    // The last property, the swapchain extent, is the
    // resolution of the swapchain images, almost always
    // exactly equal to the resolution of the window that we
    // are drawing to. There is a range of possible
    // resolutions, defined in the SurfaceCapabilitiesKHR
    // struct, with the current width and height of the surface
    // stored in the 'current_extent' field. Some window
    // managers allow different swapchain image and surface
    // resolutions, and indicate this by setting the width and
    // height in 'current_extent' to the maximum value of u32.
    // In that case, we will still pick the resolution of the
    // window, clamped between the min and max values of the
    // swapchain capabilities.
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        let size = window.inner_size();
        vk::Extent2D::builder()
            .width(size.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,)
            )
            .height(size.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ))
            .build()
    }
}

pub fn create_swapchain(
window: &Window,
instance: &Instance,
    device: &Device,
    data: &mut RenderData,
) -> Result<()> {
    // To create the swapchain, we will first query the
    // graphics queue family index and support info for the
    // device...
    let index = get_graphics_family_index(instance, data.physical_device)?;
    let support = get_swapchain_support(instance, data, data.physical_device)?;
    
    // ...as well as the image format, presentation and extent.
    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    // We then have to decide the number of images that our
    // swapchain will contain; it is recommended to have at
    // least one more than the minimum.
    let mut image_count = support.capabilities.min_image_count + 1;
    if support.capabilities.max_image_count != 0 
        && image_count > support.capabilities.max_image_count {
        image_count = support.capabilities.max_image_count;
    }

    // Then we have to decide how to handle swapchain images
    // that will be used across multiple queue families, which
    // could happen if the graphics and presentation queues are
    // different, for example. There are two possible sharing
    // modes for this:
    // - EXCLUSIVE: images are owned by one queue family at a
    //   time, and ownership must be explicitly transfered.
    //   This option offers the best performance.
    // - CONCURRENT: images can be used across multiple queue
    //   families.
    //
    // Since we only have one queue, we can use EXCLUSIVE.
    let queue_family_indices = &[index];
    let image_sharing_mode = vk::SharingMode::EXCLUSIVE;

    // We can finally fill in the (large) swapchain info
    // struct. After the swpachain image count and format, we
    // have to specify some other properties:
    // - image_array_layers: the amount of views of the image,
    //   which is always 1 except in stereoscopic applications
    //   (VR, for example, where there is always two
    //   simultaneous views of the same image)
    // - image_usage: the kind of operations we'll use the
    //   images in the swapchain for; here we're rendering to
    //   them directly, so they are used as COLOR_ATTACHMENT.
    //   It is also possible that images will be rendered
    //   separately first to perform operations like
    //   post-processing, in which case they would be used as
    //   TRANSFER_DST (transfer destination flag).
    // - pre_transform: a transform that should be applied to
    //   the images before presentation, like a clockwise
    //   rotation or horizontal flip. We don't want any special
    //   transform, so we specify the identity.
    // - composite_alpha: specifies if the alpha channel should
    //   be used for blending with other windows in the window
    //   system. We don't want that, so we set it to OPAQUE.
    // - clipped: specifies if we don't care about the color of
    //   the pixels that are obscured, for example because
    //   another window is in front of them.
    // - old_swapchain: a pointer to the prior swapchain if it
    //   is being recreated because it has become invalid or
    //   unoptimized while the application is running, for
    //   example because the window was resized.
    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_DST)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(queue_family_indices)
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    // And actually create the swapchain.
    data.swapchain = unsafe { device.create_swapchain_khr(&info, None)? };
    data.swapchain_images = unsafe { device.get_swapchain_images_khr(data.swapchain)? };
    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    info!("Swapchain created.");
    Ok(())
}

pub fn create_swapchain_image_views(
    device: &Device,
    data: &mut RenderData,
) -> Result<()> {
    // The swapchain is a structure to hold and present images.
    // In Vulkan, however, images are not used as such, but
    // under what is called an "image view", which describes
    // how to access the image and which parts of the image to
    // access. For each image in the swapchain, an image view
    // with the swapchain format is created and stored.
    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|&i| create_image_view(
            device, 
            i, 
            data.swapchain_format, 
            vk::ImageAspectFlags::COLOR,
            1,
        ))
        .collect::<Result<Vec<_>, _>>()?;

    info!("Swapchain image views created.");
    Ok(())
}

pub fn destroy_swapchain(
    device: &Device,
    data: &RenderData,
) {
    // Swapchain
    unsafe { device.destroy_swapchain_khr(data.swapchain, None) };

    // Image views
    data.swapchain_image_views
        .iter()
        .for_each(|&v| unsafe { device.destroy_image_view(v, None) });

    info!("Destroyed the swapchain and related objects.");
}