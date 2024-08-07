mod memory;
use memory::MemoryBlock;

struct Allocation {
    memory: MemoryBlock,
    offset: u64,
    size: u64,
}

struct Allocator;