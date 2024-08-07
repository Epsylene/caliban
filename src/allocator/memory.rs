use vulkanalia::vk::DeviceMemory;
use std::ffi::c_void;

pub struct MemoryBlock {
    pub memory: DeviceMemory,
    pub size: u64,
    pub mapped_ptr: *mut c_void,
}