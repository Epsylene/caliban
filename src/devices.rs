use crate::{
    app::AppData,
    queues::QueueFamilyIndices
};

use thiserror::Error;
use vulkanalia::prelude::v1_0::*;
use anyhow::{anyhow, Result};
use::log::*;

// The macro will create an error type with a Display impl that
// prints the given string.
#[derive(Error, Debug)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    // Each device has a number of associated queue families
    // that represent the supported functionalities (graphics,
    // compute shaders, transfer operations, etc.). For now, we
    // only need to check if the device supports graphics
    // operations.
    QueueFamilyIndices::get(instance, data, physical_device)?;
    
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