use std::collections::{HashMap, HashSet};

type ChunkId = u64;

struct MemoryChunk {
    id: ChunkId,
    size: usize,
    offset: usize,
    prev: Option<ChunkId>,
    next: Option<ChunkId>,
}

struct SubAllocator {
    size: usize,
    allocated: usize,
    chunks: HashMap<ChunkId, MemoryChunk>,
    free_chunks: HashSet<ChunkId>,
    id_counter: ChunkId,
}

impl SubAllocator {
    pub fn new(size: usize) -> Self {
        let id = 1;
        let mut chunks = HashMap::new();
        chunks.insert(
            id, MemoryChunk {
                id,
                size,
                offset: 0,
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
        size: usize, 
        alignment: usize
    ) -> (ChunkId, usize) {
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
                // allocation
                let offset = align_up(chunk.offset, alignment);
                todo!("granularity");

                // The aligned size is the size of the
                // allocation plus the padding due to the
                // alignment constraints
                let padding = offset - chunk.offset;
                let aligned_size = size + padding;

                // Return on the first chunk that fits the
                // aligned size
                (aligned_size <= chunk.size)
                    .then_some((chunk.id, aligned_size, offset))
            });

        // Get access to the available free chunk.
        // todo: error handling
        let (free_chunk_id, aligned_size, offset) = available_chunk.unwrap();
        let free_chunk = self.chunks.get_mut(&free_chunk_id).unwrap();

        // If the chunk is larger than the aligned size, split
        // the chunk in two parts: one for the allocation and
        // one for the remaining space.
        let (id, offset) = if free_chunk.size > aligned_size {
            let new_id = self.id_counter;
            self.id_counter += 1;

            // The new chunk starts at the free chunk offset
            // and overwrites that space up to the aligned
            // size.
            let new_chunk = MemoryChunk {
                id: new_id,
                size: aligned_size,
                offset: free_chunk.offset,
                prev: free_chunk.prev,
                next: Some(free_chunk.id),
            };

            // The remaining space of the former free chunk
            // (free_chunk.size - aligned_size) is then moved
            // after the new chunk.
            free_chunk.prev = Some(new_id);
            free_chunk.offset += aligned_size;
            free_chunk.size -= aligned_size;

            (new_chunk.id, offset)
        } else {
            // If the chunk size is exactly the aligned size
            // (it cannot be less because of the previous
            // filter), the free chunk is reclaimed and removed
            // from the free list.
            let chunk_id = free_chunk.id;
            self.free_chunks.remove(&chunk_id);

            (chunk_id, offset)
        };

        // In both cases, the allocated space has increased by
        // 'aligned_size'. We return the id of the allocated
        // chunk and its offset.
        self.allocated += aligned_size;
        (id, offset)
    }
}

fn align_up(value: usize, alignment: usize) -> usize {
    // Align the value up to the alignment: the 
    (value + alignment - 1) & !(alignment - 1)
}