use super::{suballocator::SubAllocator, Allocation};

use std::ffi::c_void;
use vulkanalia::{
    prelude::v1_0::*,
    vk::DeviceMemory,
};

pub enum MemoryLocation {
    Device,
    Shared,
}

// Each memory block represents a real piece of allocated
// memory on the device (or a shared memory), with a given
// size, and mapped to a pointer on the host. Each block has a
// sub-allocator that manages the sub-allocations within the
// block.
pub struct MemoryBlock {
    pub memory: DeviceMemory,
    pub size: u64,
    mapped_ptr: *mut c_void,
    suballocator: SubAllocator,
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

        // Allocate memory on the device
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
        
        // Create a sub-allocator for a memory chunk covering
        // the whole block.
        let suballocator = SubAllocator::new(size);

        Self {
            memory,
            size,
            mapped_ptr,
            suballocator,
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
    pub blocks: Vec<MemoryBlock>,
    pub properties: vk::MemoryPropertyFlags,
    pub memory_type: usize,
}

impl MemoryRegion {
    pub fn new(
        properties: vk::MemoryPropertyFlags,
        memory_type: usize,
    ) -> Self {
        Self {
            blocks: Vec::default(),
            properties,
            memory_type,
        }
    }

    pub fn allocate(
        &mut self, 
        device: &Device, 
        size: u64,
        alignment: u64,
    ) -> Allocation {
        // Iterate over the blocks to try to get an allocation.
        let allocation = self.blocks
            .iter_mut()
            .find_map(|block| {
                // For each block, try to sub-allocate.
                match block.suballocator.allocate(size, alignment) {
                    Ok((_, offset)) => {
                        // The mapped pointer is the pointer of the
                        // block plus the offset.
                        let mapped_ptr = unsafe { block.mapped_ptr.add(offset as usize) };
                        
                        Some(Allocation {
                            memory: block.memory,
                            offset,
                            mapped_ptr,
                        })
                    }
                    Err(_) => None,
                }
            });

        match allocation {
            Some(allocation) => allocation,
            None => {
                // If no allocation was possible (id est, all
                // blocks are full), we add a new block at the
                // end of the list and sub-allocate from it.
                let mut block = MemoryBlock::new(
                    device, 
                    size, 
                    self.memory_type
                );

                match block.suballocator.allocate(size, alignment) {
                    Ok((_, offset)) => {
                        let mapped_ptr = unsafe { block.mapped_ptr.add(offset as usize) };
                        let memory = block.memory;
                        
                        self.blocks.push(block);

                        Allocation {
                            memory,
                            offset,
                            mapped_ptr,
                        }
                    }
                    Err(_) => panic!("Failed to allocate memory."),
                }
            },
        }
    }

    fn free(&self, device: &Device) {
        // Free all allocated blocks 
        for block in self.blocks.iter() {
            block.destroy(device);
        }
    }
}