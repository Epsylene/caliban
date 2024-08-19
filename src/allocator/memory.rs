use super::{suballocator::{ChunkId, SubAllocator}, Allocation};

use std::ffi::c_void;
use anyhow::Result;
use vulkanalia::{
    prelude::v1_0::*,
    vk::DeviceMemory,
};

/// The memory location of a resource.
/// 
/// - `Device`: the resource is located on the device.
/// - `Shared`: the resource is visible by the host.
pub enum MemoryLocation {
    Device,
    Shared,
}

/// The type of resource.
/// 
/// - `Free`: the resource is not bound to any memory.
/// - `Linear`: the resource is bound to a linear memory block
///   (a buffer, for example).
/// - `NonLinear`: the resource is bound to a non-linear memory
///   block (an image with `VK_IMAGE_TILING_OPTIMAL`, for
///   example).
#[derive(Clone, Copy, PartialEq)]
pub enum ResourceType {
    Free,
    Linear,
    NonLinear,
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

        // Allocate memory on the device.
        let memory = unsafe {
            device.allocate_memory(&memory_info, None)
                .expect("Failed to allocate memory.")
        };

        // Map the memory to a pointer on the host.
        let mapped_ptr = unsafe {
            device.map_memory(
                memory, 
                0, 
                vk::WHOLE_SIZE as u64, 
                vk::MemoryMapFlags::empty()
            ).expect("Failed to map memory.")
        };
        
        // Create a sub-allocator for a set of memory chunks
        // covering the whole block.
        let suballocator = SubAllocator::new(size);

        Self {
            memory,
            size,
            mapped_ptr,
            suballocator,
        }
    }

    fn allocate(
        &mut self,
        size: u64,
        alignment: u64,
        granularity: u64,
        resource_type: ResourceType,
    ) -> Result<(ChunkId, u64)> {
        self.suballocator.allocate(size, alignment, granularity, resource_type)
    }

    fn is_empty(&self) -> bool {
        self.suballocator.allocated == 0
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
        granularity: u64,
        resource_type: ResourceType,
    ) -> Allocation {
        // Iterate over the blocks to try to get an allocation.
        let allocation = self.blocks
            .iter_mut()
            .enumerate()
            .find_map(|(idx, block)| {
                // For each block, try to sub-allocate.
                match block.allocate(size, alignment, granularity, resource_type) {
                    Ok((chunk_id, offset)) => {
                        // The mapped pointer is the pointer of
                        // the block plus the offset.
                        let mapped_ptr = unsafe { block.mapped_ptr.add(offset as usize) };
                        
                        Some(Allocation {
                            memory: block.memory,
                            offset,
                            chunk_id,
                            block_index: idx,
                            memory_type: self.memory_type,
                            mapped_ptr,
                        })
                    }
                    Err(_) => None,
                }
            });

        match allocation {
            // Sub-allocating succeeded, return the allocation.
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

                match block.allocate(size, alignment, granularity, resource_type) {
                    // If the allocation succeeded, return it.
                    Ok((chunk_id, offset)) => {
                        let mapped_ptr = unsafe { block.mapped_ptr.add(offset as usize) };
                        let memory = block.memory;
                        let memory_type = self.memory_type;
                        
                        self.blocks.push(block);
                        let block_index = self.blocks.len() - 1;

                        Allocation {
                            memory,
                            offset,
                            chunk_id,
                            memory_type,
                            block_index,
                            mapped_ptr,
                        }
                    }
                    // Else, panic (we should always be able to
                    // allocate from a new block).
                    Err(_) => panic!("Failed to allocate memory."),
                }
            },
        }
    }

    pub fn free(
        &mut self, 
        device: &Device,
        block_index: usize,
        chunk_id: ChunkId
    ) {
        // Get the block where the chunk is allocated and free
        // it.
        let block = &mut self.blocks[block_index];
        block.suballocator.free(chunk_id);
        
        // If the block is now empty, destroy it. 
        if block.is_empty() {
            block.destroy(device);
            self.blocks.remove(block_index);
        }
    }
}