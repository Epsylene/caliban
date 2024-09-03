use super::memory::{MemoryChunk, ChunkId};

/// List of free chunks.
type FreeList = Vec<MemoryChunk>;

/// Number of first level bins. The first level super-blocks go
/// from 2^4 to 2^27, so there are 27 - 4 = 23 bins.
const FL_BIN_COUNT: usize = 23;

/// Number of second level bins. We use a single byte for the
/// bitmap, so there are 8 bins, each corresponding to a range
/// 2^f(1 + n/8), where f is the first level index and n the
/// second level bin.
const SL_BIN_COUNT: usize = 8;

pub struct Tlsf {
    first_level: u32,
    second_level: [u8; FL_BIN_COUNT],
    free_lists: [[FreeList; SL_BIN_COUNT]; FL_BIN_COUNT],
}

impl Tlsf {
    pub fn new() -> Self {
        Self {
            first_level: 0,
            second_level: [0; FL_BIN_COUNT],
            free_lists: Default::default(),
        }
    }

    pub fn insert_chunk(
        &mut self, 
        chunk: MemoryChunk,
    ) {
        // Get the first and second level indices for this
        // chunk.
        let (fl, sl) = self.get_indices(chunk.size);
        self.first_level |= 1 << fl;
        self.second_level[fl] |= 1 << sl;

        // Then, insert the chunk into the corresponding free
        // list.
        self.free_lists[fl][sl].push(chunk);
    }

    pub fn get_free_chunk(
        &mut self,
        size: u64,
    ) -> Option<MemoryChunk> {
        // Since this is a good-fit strategy, we don't search
        // for a chunk with the exact same size, but take the
        // first one that has a size big enough and is in the
        // same range. Thus, we need to round the size up to
        // the next second-level block size (this may change
        // the first level index as well, which is why we
        // calculate the rounded size first).
        let rounded_size = self.next_block_size(size);
        let (fl, sl) = self.get_indices(rounded_size);
        
        // Then we can get the first free chunk from the
        // corresponding free list, which is guaranteed to be
        // large enough to fit the allocation (at the cost of
        // some extra memory).
        let chunk = self.free_lists[fl][sl].pop()?;        
        Some(chunk)
    }

    fn get_indices(&self, size: u64) -> (usize, usize) {
        // For a given chunk of size s, the first level
        // "superblock" it will be placed in is the one with
        // size 2^n <= s, so n = floor(log2(s)).
        let fl = size.ilog2() as usize;
        
        // For the second level index, blocks have sizes 2^f(1+
        // n/8) (where f is the first-level index), since each
        // bin has 8 elements. Thus, n = floor((s/2^f-1)*8).
        let sl = (size - (1 << fl)) as f32 * 8.0 / (1 << fl) as f32;
        let sl = sl.floor() as usize;

        // Return the indices, shifting fl down by 4 since we
        // start at 2^4.
        (fl-4, sl)
    }

    fn next_block_size(&self, size: u64) -> u64 {
        // Get the indices for this size (with the original
        // first level value).
        let (fl, sl) = self.get_indices(size);
        let fl = fl + 4;

        // The rounded size is the one of the second-level
        // block next to the one that contains the size, so
        // 2^fl(1 + (sl+1)/8), where (fl,sl) are first and
        // second level indices.
        ((1 << fl) as f32 * (1.0 + (sl+1) as f32/8.0)) as u64
    }
}