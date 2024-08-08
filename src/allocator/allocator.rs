use super::memory::{MemoryBlock, MemoryRegion};
use vulkanalia::prelude::v1_0::*;

struct Allocation {
    memory: MemoryBlock,
    offset: u64,
}

struct Allocator {
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    regions: Vec<MemoryRegion>,
}

impl Allocator {
    fn new(instance: Instance, physical_device: vk::PhysicalDevice) -> Self {
        let properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };

        let regions = properties.memory_types
            .iter()
            .enumerate()
            .map(|(index, memory_type)| {
                MemoryRegion::new(memory_type.property_flags, index)
            })
            .collect();
        
        Self {
            instance,
            physical_device,
            regions,
        }
    }
}