use super::memory::{MemoryLocation, MemoryBlock, MemoryRegion};
use vulkanalia::prelude::v1_0::*;

struct Allocation {
    memory: MemoryBlock,
    offset: u64,
}

struct Allocator {
    instance: Instance,
    device: Device,
    regions: Vec<MemoryRegion>,
}

impl Allocator {
    fn new(instance: Instance, device: Device, physical_device: vk::PhysicalDevice) -> Self {
        let memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };

        let regions = memory_properties.memory_types
            .iter()
            .enumerate()
            .map(|(index, memory_type)| {
                MemoryRegion::new(memory_type.property_flags, index)
            })
            .collect();
        
        Self {
            instance,
            device,
            regions,
        }
    }

    fn allocate(&mut self, requirements: vk::MemoryRequirements, location: MemoryLocation) -> Allocation {
        let memory_properties = match location {
            MemoryLocation::Device => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryLocation::Shared => vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        };

        let memory_type = self.find_memory_type(requirements, memory_properties);
        let region = &mut self.regions[memory_type];

        // todo: allocate memory from the region and return an Allocation
        // region.allocate(&self.device)
        unimplemented!()
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