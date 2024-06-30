
use crate::devices::SuitabilityError;

use vulkanalia::prelude::v1_0::*;
use anyhow::{anyhow, Result};

pub unsafe fn get_graphics_family_index(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<u32> {
    // Almost every operation in Vulkan requires commands to be
    // submitted to a queue. There are different types of
    // queues, that originate from different queue families,
    // and each family of queues allows only a subset of
    // commands. The get_physical_device_queue... function
    // contains details about the queue families supported by
    // the device.
    let queues = instance
        .get_physical_device_queue_family_properties(physical_device);

    // We can then find the first family that supports graphics
    // operations and retrieve its index.
    let graphics = queues
        .iter()
        .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|i| i as u32);

    match graphics {
        Some(index) => Ok(index),
        None => Err(anyhow!(SuitabilityError("Missing required queue families."))),
    }
}