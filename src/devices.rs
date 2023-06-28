use std::collections::HashSet;

use crate::{
    app::AppData,
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
    // devices and picking the first graphics device.
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