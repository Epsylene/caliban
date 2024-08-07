use vulkanalia::{
    prelude::v1_0::*,
    vk::DeviceMemory,
};
use std::ffi::c_void;

pub struct MemoryBlock {
    pub memory: DeviceMemory,
    pub size: u64,
    pub mapped_ptr: *mut c_void,
}

impl MemoryBlock {
    pub fn new(
        device: &Device,
        size: u64,
        memory_type: usize,
    ) -> Self {
        let memory_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(memory_type as u32);

        let memory = unsafe {
            device.allocate_memory(&memory_info, None)
                .expect("Failed to allocate memory.")
        };

        let mapped_ptr = unsafe {
            device.map_memory(
                memory, 0, 
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
}