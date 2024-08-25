use std::collections::{HashMap, HashSet};
use vulkanalia::prelude::v1_0::*;

/// Portion of memory that is sub-allocated (managed) within a
/// block.
struct MemoryChunk {
    size: u64,
    offset: u64,
}

type ChunkId = usize;

/// Memory block that is allocated from a memory region. It
/// holds one contiguous slice of `vk::DeviceMemory` and
/// sub-allocates it into chunks.
struct MemoryBlock {
    /// Actual device memory allocated from Vulkan, which is
    /// then sub-allocated into chunks.
    memory: vk::DeviceMemory,
    /// Size of the memory block.
    size: u64,
    /// List of chunks the block is comprised of.
    chunks: Vec<MemoryChunk>,
    /// The subset of chunks that are empty and can be
    /// allocated.
    free_chunks: HashSet<ChunkId>,
}

impl MemoryBlock {
    fn new(
        device: &Device,
        size: u64,
        memory_type: usize,
    ) -> Self {
        // Memory info: the block is allocated from the device
        // with a specific size and memory type.
        let memory_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(memory_type as u32);

        // Allocate memory on the device.
        let memory = unsafe {
            device.allocate_memory(&memory_info, None)
                .expect("Failed to allocate memory.")
        };

        // At first the block is empty, so it contains a single
        // chunk...
        let chunks = vec![MemoryChunk {
            size,
            offset: 0,
        }];

        // ...that is part of the free list.
        let mut free_chunks = HashSet::new();
        free_chunks.insert(0);

        // Create a new memory block with the allocated memory.
        Self {
            memory,
            size,
            chunks,
            free_chunks,
        }
    }
}