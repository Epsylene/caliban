
use crate::{app::AppData, devices::SuitabilityError};

use vulkanalia::{prelude::v1_0::*, vk::KhrSurfaceExtension};
use anyhow::{anyhow, Result};

pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub present: u32,
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

        // Then do the same for presentation, that is, that
        // there is a queue family in the device that supports
        // presenting images to a Vulkan surface (in other
        // words, rendering to a window).
        let mut present = None;
        for (index, _) in queues.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(
                physical_device,
                index as u32,
                data.surface,
            )? {
                present = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SuitabilityError("Missing required queue families.")))
        }
    }
}