use std::collections::HashMap;
use vulkanalia::prelude::v1_0::*;

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
pub struct MemoryBlock {
    /// Actual device memory allocated from Vulkan, which is
    /// then sub-allocated into chunks.
    memory: vk::DeviceMemory,
    /// Size of the memory block.
    size: u64,
    /// List of chunks the block is comprised of.
    chunks: HashMap<ChunkId, MemoryChunk>,
    /// Number of bytes currently allocated from the block.
    allocated: u64,
}

/// All blocks are allocated with a size of 256 MiB.
const MEM_BLOCK_SIZE: u64 = 256 * 1024 * 1024;

impl MemoryBlock {
    pub fn new(
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
        // block (minus one byte to avoid going out of range of
        // the TLSF structure).
        let chunk = MemoryChunk {
            size: size-1,
            offset: 0,
            prev: None,
            next: None,
        };
        let chunks = HashMap::from([(0, chunk)]);

        Self {
            memory,
            size,
            chunks,
            allocated: 0,
        }
    }

    pub fn get_chunk(&self, offset: u64) -> MemoryChunk {
        self.chunks[&offset]
    }
}

/// Memory pool blocks are allocated from. Each region
/// corresponds to a single Vulkan memory type.
pub struct MemoryRegion {
    /// List of memory blocks for linear resources.
    blocks_linear: Vec<MemoryBlock>,
    /// List of memory blocks for non-linear resources.
    blocks_non_linear: Vec<MemoryBlock>,
    /// TLSF structure to manage free chunks in linear blocks.
    free_linear: Tlsf,
    /// TLSF structure to manage free chunks in non-linear
    /// blocks.
    free_non_linear: Tlsf,
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
            blocks_linear: Vec::new(),
            blocks_non_linear: Vec::new(),
            free_linear: Tlsf::new(),
            free_non_linear: Tlsf::new(),
            properties,
            memory_type,
        }
    }

    pub fn allocate(
        &mut self,
        device: &Device,
        size: u64,
        alignment: u64,
        resource_type: ResourceType,
    ) -> Allocation {
        // Linear and non-linear resources are managed
        // independently, in order to avoid having to deal with
        // granularity.
        let (tlsf, blocks) = match resource_type {
            ResourceType::Linear => (&mut self.free_linear, &mut self.blocks_linear),
            ResourceType::NonLinear => (&mut self.free_non_linear, &mut self.blocks_non_linear),
        };

        // Request a free chunk to allocate from.
        let (block, offset) = match tlsf.get_free_chunk(size) {
            Some(chunk) => {
                // If a free chunk was found, return its block
                // and its offset.
                (chunk.block, chunk.offset)
            }
            None => {
                // Else, there is no free space available, so
                // we first need to create a new memory block.
                blocks.push(MemoryBlock::new(
                    device,
                    MEM_BLOCK_SIZE,
                    self.memory_type,
                ));

                // The block is the last of the list; it is of
                // course empty, so it contains a single free
                // chunk at offset 0.
                let block = blocks.len()-1;
                let offset = 0;

                tlsf.insert_chunk(
                    MEM_BLOCK_SIZE-1,
                    offset,
                    block,
                );

                (block, offset)
            }
        };

        // The offset must be aligned to the value given by the
        // memory requirements.
        let offset = align_up(offset, alignment);
        
        // The chunk is now in place, so we can return the
        // offset and the memory handle of the block.
        Allocation {
            memory: blocks[block].memory,
            offset,
        }
    }
}

fn align_down(value: u64, alignment: u64) -> u64 {
    // Align a value down to another value (the alignment): let
    // us take for example V = 0x3F and an alignment A = 0x20.
    // We have:
    // 
    //  A = 0010 0000
    //  A - 1 = 0001 1111 (set all lower bits)
    //  M = !(A-1) = 1110 0000 (invert to get a mask of the
    //                          higher bits)
    //  
    //    V = 0011 1111
    //  & M = 1110 0000
    //  ---------------
    //        0010 0000
    //
    // All bits of V higher than A have been set to 0, so in
    // the end align_down(V) = 0x20 = A. If we had V = 0x1F =
    // 0001 1111, we would have end up with align_down(V) =
    // 0x0, which is the next lower value that is a multiple of
    // the page size A.
    value & !(alignment - 1)
}

fn align_up(value: u64, alignment: u64) -> u64 {
    // Aligning up is aligning down the value shifted by one
    // page (that is, value + alignment - 1).
    align_down(value + alignment - 1, alignment)
}