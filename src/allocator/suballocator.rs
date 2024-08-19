use super::memory::ResourceType;

use std::collections::{HashMap, HashSet};
use anyhow::{anyhow, Result};

pub type ChunkId = u64;

struct MemoryChunk {
    id: ChunkId,
    size: u64,
    offset: u64,
    resource_type: ResourceType,
    prev: Option<ChunkId>,
    next: Option<ChunkId>,
}

pub struct SubAllocator {
    size: u64,
    chunks: HashMap<ChunkId, MemoryChunk>,
    free_chunks: HashSet<ChunkId>,
    id_counter: ChunkId,
    pub allocated: u64,
}

impl SubAllocator {
    pub fn new(size: u64) -> Self {
        let id = 1;
        let mut chunks = HashMap::new();

        // Initialize with a single free chunk that covers the
        // whole memory block.
        chunks.insert(
            id, MemoryChunk {
                id,
                size,
                offset: 0,
                resource_type: ResourceType::Free,
                prev: None,
                next: None,
            }
        );

        let mut free_chunks = HashSet::new();
        free_chunks.insert(id);

        Self {
            size,
            allocated: 0,
            chunks,
            free_chunks,
            id_counter: id + 1,
        }
    }

    pub fn allocate(
        &mut self, 
        size: u64, 
        alignment: u64,
        granularity: u64,
        resource_type: ResourceType,
    ) -> Result<(ChunkId, u64)> {
        // Find a free chunk that fits the allocation
        // constraints (size and alignment).
        let available_chunk = self.free_chunks
            .iter()
            .filter_map(|id| {
                // Get a chunk that is large enough to fit the
                // allocation
                let chunk = self.chunks.get(id).unwrap();
                (chunk.size >= size).then_some(chunk)
            })
            .find_map(|chunk| {
                // Get the correctly aligned offset for the
                // allocation.
                let mut offset = align_up(chunk.offset, alignment);
                
                // If there is a previous chunk...
                if let Some(prev_id) = chunk.prev {
                    let prev = self.chunks.get(&prev_id).unwrap();

                    // ...check if it is on the same page as
                    // the current one and if the resource to
                    // be allocated conflicts with the previous
                    // one.
                    if is_on_same_page(prev.offset, prev.size, offset, size)
                        && granularity_conflict(prev.resource_type, resource_type) {
                        // In that case, align the offset to
                        // the granularity.
                        offset = align_up(offset, granularity);
                    }
                }

                // The aligned size is the size of the
                // allocation plus the padding due to the
                // alignment constraints
                let padding = offset - chunk.offset;
                let aligned_size = size + padding;

                if let Some(next) = chunk.next {
                    let next_chunk = self.chunks.get(&next).unwrap();

                    // If the next chunk also conflicts,
                    // return: we cannot allocate here.
                    if is_on_same_page(offset, size, next_chunk.offset, next_chunk.size)
                        && granularity_conflict(resource_type, next_chunk.resource_type) {
                        return None;
                    }
                }

                // Return on the first chunk that fits the
                // aligned size
                (aligned_size <= chunk.size)
                    .then_some((chunk.id, aligned_size, offset))
            });

        // Get access to the available free chunk. If there is
        // none, return early.
        let (free_chunk_id, aligned_size, offset) = match available_chunk {
            Some(chunk) => chunk,
            None => return Err(anyhow!("No free chunk available")),
        };
        let free_chunk = self.chunks.get_mut(&free_chunk_id).unwrap();

        // If the chunk is larger than the aligned size, split
        // the chunk in two parts: one for the allocation and
        // one for the remaining space.
        let id = if free_chunk.size > aligned_size {
            let new_id = self.id_counter;
            self.id_counter += 1;

            // The new chunk starts at the free chunk offset
            // and overwrites that space up to the aligned
            // size.
            let new_chunk = MemoryChunk {
                id: new_id,
                size: aligned_size,
                offset: free_chunk.offset,
                resource_type,
                prev: free_chunk.prev,
                next: Some(free_chunk.id),
            };

            // The remaining space of the former free chunk
            // (free_chunk.size - aligned_size) is then moved
            // after the new chunk.
            free_chunk.prev = Some(new_id);
            free_chunk.offset += aligned_size;
            free_chunk.size -= aligned_size;

            // If there was a previous chunk, update its 'next'
            // field too. 
            if let Some(prev_id) = new_chunk.prev {
                let prev_chunk = self.chunks.get_mut(&prev_id).unwrap();
                prev_chunk.next = Some(new_id);
            }

            // Finally, insert the new chunk in the list and
            // return the id.
            self.chunks.insert(new_id, new_chunk);
            new_id
        } else {
            // If the chunk size is exactly the aligned size
            // (it cannot be less because of the previous
            // filter), the free chunk is reclaimed and removed
            // from the free list.
            let chunk_id = free_chunk.id;
            self.free_chunks.remove(&chunk_id);

            chunk_id
        };

        // In both cases, the allocated space has increased by
        // 'aligned_size'. We return the id of the allocated
        // chunk and its offset.
        self.allocated += aligned_size;
        Ok((id, offset))
    }

    pub fn free(&mut self, chunk_id: ChunkId) {
        let chunk = self.chunks.get_mut(&chunk_id).unwrap();

        chunk.prev = None;
        chunk.next = None;
        self.allocated -= chunk.size;
        self.free_chunks.insert(chunk_id);
    }

    fn merge_chunks(&mut self, chunk_l: ChunkId, chunk_r: ChunkId) {
        // Get the right chunk and remove it from the list,
        // since it will be merged.
        let chunk_right = self.chunks.remove(&chunk_r).unwrap();
        self.free_chunks.remove(&chunk_right.id);
        
        // Get the left chunk and update its size and `next`
        // pointer.
        let chunk_left = self.chunks.get_mut(&chunk_l).unwrap();
        chunk_left.size += chunk_right.size;
        chunk_left.next = chunk_right.next;

        // Get the 'next' chunk of the (merged) right chunk and
        // update its `prev` pointer.
        if let Some(next_id) = chunk_right.next {
            let next_chunk = self.chunks.get_mut(&next_id).unwrap();
            next_chunk.prev = Some(chunk_l);
        }
    }
}

fn align_down(value: u64, alignment: u64) -> u64 {
    // Align a value down to another value (the alignment): let
    // us take an example with a value V = 0x3F and an alignment
    // of A = 0x20. We have:
    //  A = 0010 0000
    //  A - 1 = 0001 1111 (set all bits lower than A)
    //  M = !(A-1) = 1110 0000 (invert to get a mask)
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
    // Aligning up is aligning down the value offset by one
    // page (alignment - 1).
    align_down(value + alignment - 1, alignment)
}

fn is_on_same_page(
    offset_a: u64, 
    size_a: u64, 
    offset_b: u64, 
    granularity: u64
) -> bool {
    // The specification requires that linear and non-linear
    // resources be placed on separate pages of size
    // `granularity`. The end of the page the first object (A)
    // is allocated on is the actual end of the object (its
    // offset plus the length) down-aligned to the page size
    // (granularity). Then, the start of the second object's
    // page (B) is the start of the object (the offset) aligned
    // down to the granularity.
    let end_a = offset_a + size_a - 1;
    let end_page_a = align_down(end_a, granularity);
    let start_page_b = align_down(offset_b, granularity);

    // The two objects are on the same page if the end of the
    // first page overlaps the start of the second page.
    end_page_a >= start_page_b
}

fn granularity_conflict(
    type_a: ResourceType, 
    type_b: ResourceType
) -> bool {
    // If one of the two memory chunks has a "free" resource
    // (i.e., it is not currently allocated), there is no
    // conflict.
    if type_a == ResourceType::Free || type_b == ResourceType::Free {
        return false;
    }

    // Otherwise, there is a conflict if the two resources are
    // not of the same type (that is, if one is linear and the
    // other non-linear).
    type_a != type_b
}