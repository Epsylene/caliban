
use crate::{app::AppData, devices::SuitabilityError};

use vulkanalia::prelude::v1_0::*;
use anyhow::{anyhow, Result};

pub struct QueueFamilyIndices {
    pub graphics: u32,
}

impl QueueFamilyIndices {
    pub unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        // Almost every operation in Vulkan requires commands to
        // be submitted to a queue. There are different types of
        // queues, that originate from different queue families,
        // and each family of queues allows only a subset of
        // commands. The get_physical_device_queue... function
        // contains details about the queue families supported
        // by the device.
        let queues = instance
            .get_physical_device_queue_family_properties(physical_device);

        // We can then find the first family that supports
        // graphics operations and retrieve its index.
        let graphics = queues
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        if let Some(graphics) = graphics {
            Ok(Self { graphics })
        } else {
            Err(anyhow!(SuitabilityError("Missing required queue families.")))
        }
    }
}