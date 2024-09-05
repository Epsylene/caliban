mod memory;
mod tlsf;

use vulkanalia::prelude::v1_0::*;
use memory::{MemoryUse, ResourceType, MemoryRegion};

/// A memory allocation object, that holds the information
/// necessary to bind a resource to Vulkan memory.
pub struct Allocation {
    /// The Vulkan device memory object the allocation is tied to.
    pub memory: vk::DeviceMemory,
    /// The offset of the allocation within the memory object.
    pub offset: u64,
}

/// Memory allocator that manages Vulkan memory and provides
/// functions to allocate and free resources from it.
pub struct Allocator {
    /// Memory regions that are supported by the device. Each
    /// memory region corresponds to a single Vulkan memory
    /// type.
    regions: Vec<MemoryRegion>,
}

impl Allocator {
    pub fn new(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Self {
        // Get the memory properties of the device.
        let memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };

        // Then, create a memory region for each memory type
        // supported by the device. The region registers the
        // property flags and the index of the memory type.
        let regions = memory_properties.memory_types
            .iter()
            .enumerate()
            .map(|(index, memory)| {
                MemoryRegion::new(index, memory.property_flags)
            })
            .collect();

        Self {
            regions,
        }
    }

    pub fn allocate(
        &mut self, 
        device: &Device,
        requirements: vk::MemoryRequirements, 
        location: MemoryUse,
        resource_type: ResourceType,
    ) -> Allocation {
        // Request memory properties based on the desired use:
        // for a gpu-only memory, we only need to set the
        // DEVICE_LOCAL flag, while for data transfered between
        // the host to the device, we want to set the
        // DEVICE_LOCAL and HOST_VISIBLE flags.
        let requested_properties = match location {
            MemoryUse::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryUse::CpuToGpu => vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
        };

        // Find the memory type that satisfies the requirements
        // and properties, and select the region corresponding
        // to this memory type.
        let memory_type = self.find_memory_type(requirements, requested_properties);
        let region = &mut self.regions[memory_type];

        // Then, allocate a memory block from the region and
        // return the allocation.
        region.allocate(
            device,
            requirements.size,
            requirements.alignment,
            resource_type,
        )
    }

    fn find_memory_type(&self, requirements: vk::MemoryRequirements, properties: vk::MemoryPropertyFlags) -> usize {
        // Find a memory type that is suitable for the buffer
        // with the given requirements and properties. Each
        // memory region corresponds to a memory type index, so
        // we just need to find the right one and return the
        // index.
        self.regions
            .iter()
            .find(|region| {
                let type_index = &region.memory_type;
                let memory_properties = &region.properties;

                // The "memory type bits" field of the
                // requirements has a bit set at the index of
                // the required memory type, so we mask with
                // the region's memory index. Furthermore, the
                // region memory properties must contain the
                // required properties.
                requirements.memory_type_bits & (1 << type_index) != 0
                    && memory_properties.contains(properties)
            })
            .map(|region| region.memory_type)
            .expect("Failed to find suitable memory type.")
    }
}