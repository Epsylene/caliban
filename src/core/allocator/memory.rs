use std::collections::HashMap;
use vulkanalia::prelude::v1_0::*;
use anyhow::Result;

use super::Allocation;
use super::tlsf::Tlsf;

/// How a memory resource will be used.
pub enum MemoryUse {
    /// Resource that is only used by the GPU. Corresponds to
    /// the `DEVICE_LOCAL` flag.
    GpuOnly,
    /// Resource that is uploaded from the CPU to the GPU.
    /// Corresponds to `DEVICE_LOCAL | HOST_VISIBLE`.
    CpuToGpu,
}

/// Type of the resource to be allocated.
pub enum ResourceType {
    /// The resource is bound to a linear memory block (a
    /// buffer, for example).
    Linear,
    /// The resource is bound to a non-linear memory block (an
    /// image with `VK_IMAGE_TILING_OPTIMAL`, for example).
    NonLinear,
}

/// Portion of memory that is sub-allocated (managed) within a
/// block.
#[derive(Clone, Copy)]
pub struct MemoryChunk {
    /// Size of the chunk in bytes.
    pub size: u64,
    /// Offset of the chunk within the memory block.
    pub offset: u64,
    /// Index of the previous chunk in the block.
    pub prev: Option<ChunkId>,
    /// Index of the next chunk in the block.
    pub next: Option<ChunkId>,
}

/// Unique identifier of a chunk within a memory block. This is
/// in fact just the offset of the chunk within the block.
pub type ChunkId = u64;

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
    chunks: HashMap<ChunkId, MemoryChunk>,
    /// The subset of chunks that are empty and can be
    /// allocated.
    free_chunks: Tlsf,
    /// Number of bytes currently allocated from the block.
    allocated: u64,
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
        // chunk at offset 0 that spans the entire size of the
        // block...
        let chunk = MemoryChunk {
            size,
            offset: 0,
            prev: None,
            next: None,
        };
        let chunks = HashMap::from([(0, chunk)]);

        // ...and is not in use, so part of the free list. The
        // "ID" of a chunk is given by its offset, since it is
        // sufficient to identify the chunk within the block.
        let mut free_chunks = Tlsf::new();
        free_chunks.insert_chunk(chunk);

        Self {
            memory,
            size,
            chunks,
            free_chunks,
            allocated: 0,
        }
    }
}

/// Memory pool blocks are allocated from. Each region
/// corresponds to a single Vulkan memory type.
pub struct MemoryRegion {
    /// List of memory blocks that are allocated from the
    /// region.
    blocks: Vec<MemoryBlock>,
    /// Index of the memory type of the region.
    pub memory_type: usize,
    /// Properties of the memory type of the region.
    pub properties: vk::MemoryPropertyFlags,
}

impl MemoryRegion {
    pub fn new(
        memory_type: usize,
        properties: vk::MemoryPropertyFlags,
    ) -> Self {
        Self {
            blocks: Vec::new(),
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
    ) -> Result<Allocation> {
        todo!()
    }
}