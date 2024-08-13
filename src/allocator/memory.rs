use super::Allocation;
use std::ffi::c_void;
use vulkanalia::{
    prelude::v1_0::*,
    vk::DeviceMemory,
};

pub enum MemoryLocation {
    Device,
    Shared,
}

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
    pub blocks: Vec<Option<MemoryBlock>>,
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

    pub fn allocate(&mut self, device: &Device, size: usize) -> Allocation {
        // Get the index of the first empty (fully
        // non-allocated) block.
        let index = self.blocks
            .iter()
            .enumerate()
            .find_map(|(index, block)| {
                if block.is_none() {
                    // If there is no block, we can allocate,
                    // so return the index.
                    Some(index)
                } else {
                    // If there is a block, try to allocate
                    // with the sub-allocator.
                    todo!();

                    // If the sub-allocation succeeds, get the
                    // offset from the block's mapped pointer
                    // and return the allocation.
                    todo!();

                    None
                }
            });

        // If there was a fully empty block or no
        // sub-allocation suceeded, allocate a block of memory
        // from the device.
        let block = MemoryBlock::new(
            device, 
            size as u64, 
            self.memory_type
        );
        
        // Match the index depending on the previous case:
        let index = match index {
            // - If there was an empty block, place the new
            //   block at that index.
            Some(index) => {
                self.blocks[index] = Some(block);
                index
            },
            // - If all blocks were full, push the new block to
            //   the end of the list.
            None => {
                self.blocks.push(Some(block));
                self.blocks.len() - 1
            }
        };

        // Sub-allocate the memory from the block and get the
        // mapped pointer with the offset.
        todo!();

        // Return the allocation.
        todo!();
    }

    fn free(&self, device: &Device) {
        // Free all allocated blocks 
        for block in self.blocks.iter().flatten() {
            block.destroy(device);
        }
    }
}