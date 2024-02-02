use std::collections::HashSet;

use crate::{
    app::{AppData, VALIDATION_ENABLED, VALIDATION_LAYER, PORTABILITY_MACOS_VERSION},
    queues::QueueFamilyIndices,
    swapchain::SwapchainSupport,
};

use thiserror::Error;
use vulkanalia::prelude::v1_0::*;
use anyhow::{anyhow, Result};
use::log::*;

pub const REQUIRED_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

// The macro will create an error type with a Display impl that
// prints the given string.
#[derive(Error, Debug)]
#[error("{0}")]
pub struct SuitabilityError(pub &'static str);

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    // Get the list of supported device extensions on the device
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();

    // Check if all required extensions are supported
    if REQUIRED_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError("Missing required device extensions.")))
    }
}

unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    // Each device has a number of associated queue families
    // that represent the supported functionalities (graphics,
    // compute shaders, transfer operations, etc.). Those are
    // retrieved and necessary operations are checked for.
    QueueFamilyIndices::get(instance, data, physical_device)?;
    
    // Then we can check if the device supports all the required
    // extensions.
    check_physical_device_extensions(instance, physical_device)?;

    // Likewise, we can check if the device supports the
    // included optional features.
    let features = instance.get_physical_device_features(physical_device);
    if features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(SuitabilityError("Device does not support anisotropic filtering.")));
    }
    
    // Finally, we can check if the device's swapchain support
    // is sufficient. We want to at least have one supported
    // image format and presentation mode for our window
    // surface.
    let support = SwapchainSupport::get(instance, data, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }

    Ok(())
}

pub unsafe fn pick_physical_device(
    instance: &Instance, 
    data: &mut AppData
) -> Result<()> {
    // There can be more than one graphics device on the system
    // (one dedicated and one integrated graphics card at the
    // same time, for example), and in fact a Vulkan instance
    // can set up and use any number of them simultaneously, but
    // we will stick here to listing the available physical
    // devices and picking the first graphics-capable one.
    for device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(device);

        if let Err(error) = check_physical_device(instance, data, device) {
            warn!("Skipping physical device ({}): {}", properties.device_name, error);
        } else {
            info!("Selected physical device: {}", properties.device_name);
            data.physical_device = device;
            return Ok(());
        }
    }

    Err(anyhow!(SuitabilityError("Failed to find suitable physical device.")))
}

pub unsafe fn create_logical_device(
    entry: &Entry, 
    instance: &Instance, 
    data: &mut AppData,
) -> Result<Device> {
    // The logical device serves as a layer between a physical
    // device and the application. There might be more than one
    // logical device per physical device, each representing
    // different sets of requirements. To create the logical
    // device, we need to build a representation of the queue
    // families of the physical device we are using, and in
    // particular to specify the number of queues for each queue
    // family; most drivers will only allow for a small number
    // of queues per family, but that is not a problem since the
    // command buffers can be created on multiple threads and
    // submitted all at once with minimal overhead. To build the
    // queue family info, we first need to get the indices of
    // the physical device queue families.
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    
    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    // We can then build the queue families info struct. For
    // each supported queue family in our device, we are
    // providing its index on the device to Vulkan and building
    // the set of associated queues with their priorities (that
    // is, a number between 0.0 and 1.0 which influences the
    // scheduling of command buffers execution); since we only
    // want one queue per family, but are still required to
    // provide the priorities, we simply input the array [1.0].
    let priorities = &[1.0];
    let queues = unique_indices
        .iter()
        .map(|&index| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(index)
                .queue_priorities(priorities)
                .build()
        })
        .collect::<Vec<_>>();

    // The next piece of information for the logical devices are
    // layers and extensions. Previous implementations of Vulkan
    // made a distinction between instance and device specific
    // validation layers, but this is no longer the case.
    // However, it is still a good idea to set them anyway to be
    // compatible with older implementations.
    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    // Then we add the required extensions.
    let mut extensions = REQUIRED_EXTENSIONS
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    // Some implementations are not fully conformant, so certain
    // Vulkan extensions need to be enabled to ensure
    // portability.
    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
    }

    // We can then specify the set of optional device features
    // we want to have, such as anisotropic filtering. 
    let features = vk::PhysicalDeviceFeatures::builder()
        .sampler_anisotropy(true);

    // The device info struct combines all the information we
    // have gathered so far.
    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queues)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    // Finally, we can create the device, and set our app handles
    // for the graphics and presentation queues.
    let device = instance.create_device(data.physical_device, &info, None)?;
    
    data.graphics_queue = device.get_device_queue(indices.graphics, 0);
    data.present_queue = device.get_device_queue(indices.present, 0);

    info!("Logical device created.");
    Ok(device)
}