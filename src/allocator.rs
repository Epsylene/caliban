mod memory;
mod suballocator;

use memory::{MemoryLocation, MemoryRegion, ResourceType};
use suballocator::ChunkId;

use vk::DeviceMemory;
use vulkanalia::prelude::v1_0::*;
use std::ffi::c_void;

pub struct Allocation {
    memory: DeviceMemory,
    offset: u64,
    chunk_id: ChunkId,
    block_index: usize,
    memory_type: usize,
    mapped_ptr: *mut c_void,
}

pub struct Allocator {
    regions: Vec<MemoryRegion>,
    granularity: u64,
}

impl Allocator {
    pub fn new(instance: Instance, device: Device, physical_device: vk::PhysicalDevice) -> Self {
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
            .map(|(index, memory_type)| {
                MemoryRegion::new(memory_type.property_flags, index)
            })
            .collect();

        let device_properties = unsafe {
            instance.get_physical_device_properties(physical_device)
        };

        // The granularity of the device is a secondary
        // alignement constraint that must be satisfied when
        // linear and non-linear resources are placed
        // contiguously in memory.
        let granularity = device_properties.limits.buffer_image_granularity;
        
        Self {
            regions,
            granularity,
        }
    }

    fn allocate(
        &mut self, 
        device: &Device,
        requirements: vk::MemoryRequirements, 
        location: MemoryLocation,
        resource_type: ResourceType,
    ) -> Allocation {
        // Determine the memory properties based on the desired
        // location: for a device-local memory, we only need to
        // set the DEVICE_LOCAL flag, while for data shared
        // between the host and the device, we need to set the
        // HOST_VISIBLE and HOST_COHERENT flags.
        let memory_properties = match location {
            MemoryLocation::Device => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryLocation::Shared => vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        };

        // Find the memory type that satisfies the requirements
        // and properties, and select the region corresponding
        // to this memory type.
        let memory_type = self.find_memory_type(requirements, memory_properties);
        let region = &mut self.regions[memory_type];

        // Then, allocate a memory block from the region and
        // return the allocation.
        region.allocate(
            device,
            requirements.size,
            requirements.alignment,
            self.granularity,
            resource_type,
        )
    }

    fn free(&mut self, allocation: Allocation, device: &Device) {
        // Get the region corresponding to the memory type of
        // the allocation, and free the chunk in the block
        // corresponding to the allocation.
        let region = &mut self.regions[allocation.memory_type];
        region.free(device, allocation.block_index, allocation.offset);
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