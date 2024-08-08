use std::ffi::c_void;
use vulkanalia::{
    prelude::v1_0::*,
    vk::DeviceMemory,
};

pub struct MemoryBlock {
    pub memory: DeviceMemory,
    pub size: u64,
    pub mapped_ptr: *mut c_void,
}

impl MemoryBlock {
    fn new(
        device: &Device,
        size: u64,
        memory_type: usize,
    ) -> Self {
        let memory_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(memory_type as u32);

        // Memory allocated on the device
        let memory = unsafe {
            device.allocate_memory(&memory_info, None)
                .expect("Failed to allocate memory.")
        };

        // Map the memory to a pointer on the host
        let mapped_ptr = unsafe {
            device.map_memory(
                memory, 
                0, 
                vk::WHOLE_SIZE as u64, 
                vk::MemoryMapFlags::empty()
            ).expect("Failed to map memory.")
        };
        
        Self {
            memory,
            size,
            mapped_ptr,
        }
    }

    fn destroy(&self, device: &Device) {
        unsafe {
            if !self.mapped_ptr.is_null() {
                device.unmap_memory(self.memory);
            }
            device.free_memory(self.memory, None);
        }
    }
}

pub struct MemoryRegion {
    blocks: Vec<MemoryBlock>,
    properties: vk::MemoryPropertyFlags,
    type_index: usize,
}

impl MemoryRegion {
    pub fn new(
        properties: vk::MemoryPropertyFlags,
        type_index: usize,
    ) -> Self {
        Self {
            blocks: Vec::default(),
            properties,
            type_index,
        }
    }

    fn free(&self, device: &Device) {
        for block in self.blocks.iter() {
            block.destroy(device);
        }
    }
}