/// Chunk metadata used by the TLSF allocator.
pub struct ChunkInfo {
    /// Size of the chunk in bytes.
    size: u64,
    /// Offset of the chunk within the memory block.
    pub offset: u64,
    /// Index of the block the chunk is part of.
    pub block: usize,
}

/// List of free chunks.
type FreeList = Vec<ChunkInfo>;

/// Number of first level bins. The first level super-blocks go
/// from 2^4 (16 b) to 2^27 (128 Mb), so there are 27 - 4 = 23
/// bins.
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
        size: u64,
        offset: u64,
        block: usize,
    ) {
        // Set the bits corresponding to the first and second
        // level for this chunk.
        let (fl, sl) = self.get_indices(size);
        self.first_level |= 1 << fl;
        self.second_level[fl] |= 1 << sl;

        // Then, insert the chunk into the corresponding free
        // list.
        self.free_lists[fl][sl].push(
            ChunkInfo {
                size,
                offset,
                block,
            }
        );
    }

    pub fn get_free_chunk(
        &mut self,
        size: u64,
    ) -> Option<ChunkInfo> {
        // The good-fit strategy doesn't search for a chunk
        // with the exact same size, but the first available
        // one that is large enough to fit the allocation. Note
        // that this is still O(1), since the bitmaps are fixed
        // size.
        let (fl, sl) = self.find_available(size)?;
        let chunk = self.free_lists[fl][sl].pop()?;

        // The minimum size of this free chunk is the size of
        // the allocation rounded up to the next second level
        // block size, since that is where we start looking for
        // free chunks.
        let minimum_size = self.next_block_size(size);
        
        // Then, the remaining free space is re-inserted back
        // into the TLSF structure, if it is large enough.
        let remainder = chunk.size - minimum_size;
        if remainder > 16 {
            let offset = chunk.offset + minimum_size;
            self.insert_chunk(remainder, offset, chunk.block);
        }

        Some(chunk)
    }

    fn find_available(
        &self,
        size: u64,
    ) -> Option<(usize, usize)> {
        // Get the first and second level indices for this
        // size.
        let (start_fl, start_sl) = self.get_indices(size);

        // To find the first available chunk, we start by
        // checking the second level blocks, starting from the
        // one after that of the current size (chunks of the
        // same block might be smaller than the requested
        // size).
        let sl = self.second_level[start_fl] & (!0 << (start_sl+1));
        
        if sl == 0 {
            // If no second level blocks in the current superblock
            // are available, we have to keep searching, starting
            // from the next superblock.
            for fl in (start_fl+1)..FL_BIN_COUNT {
                if self.first_level & (1 << fl) != 0 {
                    // If a first level bit is set, we know
                    // that any second level block is already
                    // large enouggh to fit the allocation, so
                    // we return the first one available.
                    let sl = self.second_level[fl].trailing_zeros() as usize;
                    return Some((fl, sl));
                }
            }

            // If no superblock is available, return None.
            None
        } else {
            // If one (or several) blocks are available in the
            // current superblock, return the first one.
            let sl = sl.trailing_zeros() as usize;
            Some((start_fl, sl))
        }
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
        // Get the indices for this size (with the actual first
        // level index).
        let (fl, sl) = self.get_indices(size);
        let fl = fl + 4;

        // The rounded size is that of the second-level block
        // next to the current one, so 2^fl(1 + (sl+1)/8),
        // where (fl,sl) are first and second level indices.
        ((1 << fl) as f32 * (1.0 + (sl+1) as f32/8.0)) as u64
    }
}