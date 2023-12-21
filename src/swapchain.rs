use crate::{app::AppData, queues::QueueFamilyIndices};

use winit::window::Window;
use vulkanalia::{
    prelude::v1_0::*,
    vk::KhrSurfaceExtension,
    vk::KhrSwapchainExtension,
};
use anyhow::Result;
use log::info;

pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    pub unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        // There is no concept of a "default framebuffer" in
        // Vulkan as there is in OpenGL, so it requires an
        // infrastructure that will own the buffers we will
        // render to before we visualize them on the screen.
        // This is the swapchain, essentially a queue of images
        // that are waiting to be presented to the screen. Not
        // all graphics cards are capable of presenting images
        // directly to a screen (for example because they are
        // designed for servers and don't have any display
        // outputs), so swapchain support and compatibility with
        // our window surface have to be queried beforehand.
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(
                    physical_device,
                    data.surface,
                )?,
            formats: instance
                .get_physical_device_surface_formats_khr(
                    physical_device,
                    data.surface,
                )?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(
                    physical_device,
                    data.surface,
                )?,
        })
    }
}

fn get_swapchain_surface_format(
    formats: &[vk::SurfaceFormatKHR],
) -> vk::SurfaceFormatKHR {
    // The first setting to determine is the surface format,
    // which itself consists of two fields: 'format', which
    // specifies the color channels and types, and 'color_space'
    // which indicates the supported color space. In our case,
    // we will want a B8G8R8A8_SRGB format (B, G, R and alpha
    // channels of 8 bits each in sRGB color space, which makes
    // for 32 bits of color per pixel, the most common bit
    // depth) and a sRGB color space (standard non-linear RGB
    // space, made to match more closely the way the human eye
    // perceives color). If this surface format is not
    // available, we will just default on the first one
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
    // resolution of the swapchain images, almost always exactly
    // equal to the resolution of the window that we are drawing
    // to. There is a range of possible resolutions, defined in
    // the SurfaceCapabilitiesKHR struct, with the current width
    // and height of the surface stored in the 'current_extent'
    // field. Some window managers allow different swapchain
    // image and surface resolutions, and indicate this by
    // setting the width and height in 'current_extent' to the
    // maximum value of u32. In that case, we will still pick
    // the resolution of the window, clamped between the min and
    // max values of the swapchain capabilities.
    if capabilities.current_extent.width != u32::max_value() {
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

pub unsafe fn create_swapchain_image_views(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // The swapchain is a structure to hold and present images.
    // In Vulkan, however, images are not used as such, but
    // under what is called an "image view", which describes how
    // to access the image and which parts of the image to
    // access.
    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|&i| {
            // The first element of the view to define is how
            // the image colors are mapped to the image view
            // colors. We don't want to swizzle (map to a value)
            // the color components here, so we just go for the
            // identity.
            let component_mapping = vk::ComponentMapping::builder()
                .r(vk::ComponentSwizzle::IDENTITY)
                .g(vk::ComponentSwizzle::IDENTITY)
                .b(vk::ComponentSwizzle::IDENTITY)
                .a(vk::ComponentSwizzle::IDENTITY)
                .build();

            // The next thing to consider is the subresource
            // range, which describes the image's purpose and
            // which parts of the image should be accessed,
            // among other things:
            // - aspect_mask: the part of the image data (the
            //   image aspect) to be accessed (one image could
            //   contain both RGB data and depth info bits, for
            //   example). In this case, we want the color bits
            //   of the image;
            // - base_mip_level: the first accessible mipmap
            //   level (here 0, since we don't use mipmapping
            //   yet);
            // - level_count: the number of accessible mipmap
            //   levels;
            // - base_array_layer: the first accessible array
            //   layer (simultaneous views of the same image,
            //   for example for stereoscopic rendering);
            // - layer_count: the number of accessible array
            //   layers.
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build();

            // Then we can build the info struct, containing the
            // image itself, the view type of the image (1D, 2D,
            // or 3D texture, cube map), its format, the mapping
            // of the color components between the image and the
            // view, and the subresource range of the image
            // view.
            let info = vk::ImageViewCreateInfo::builder()
                .image(i)
                .view_type(vk::ImageViewType::_2D)
                .format(data.swapchain_format)
                .components(component_mapping)
                .subresource_range(subresource_range);

            device.create_image_view(&info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    info!("Swapchain image views created.");
    Ok(())
}

pub unsafe fn create_swapchain(
window: &Window,
instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // To create the swapchain, we will first query the queue
    // family indices and support struct for the device...
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, data, data.physical_device)?;
    // ...with the image format, presentation and extent.
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
    //   time, and ownership must be explicitly transfered. This
    //   option offers the best performance.
    // - CONCURRENT: images can be used across multiple queue
    //   families.
    //
    // We will use the concurrent mode if the graphics and
    // presentation families are different to keep things simple,
    // but on most hardware those two are the same.
    let mut queue_family_indices = vec![];
    let sharing_mode = if indices.graphics != indices.present {
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    // We can finally fill in the (large) swapchain info struct.
    // After the swpachain image count and format, we have to
    // specify some other properties:
    // - image_array_layers: the amount of views of the image,
    //   which is always 1 except in stereoscopic applications
    //   (VR, for example, where there is always two
    //   simultaneous views of the same image)
    // - image_usage: the kind of operations we'll use the
    //   images in the swapchain for; here we're rendering to
    //   them directly, so they are used as COLOR_ATTACHMENT. It
    //   is also possible that images will be rendered
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
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    // And actually create the swapchain.
    data.swapchain = device.create_swapchain_khr(&info, None)?;
    data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;
    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    info!("Swapchain created.");
    Ok(())
}