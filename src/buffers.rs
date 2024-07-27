use crate::{
    renderer::RenderData,
    devices::SuitabilityError, 
};

use vulkanalia::prelude::v1_0::*;
use anyhow::{Result, anyhow};

pub unsafe fn find_memory_type(
    instance: &Instance,
    data: &RenderData,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Result<u32> {
    // Graphics cards can offer different types of memory to
    // allocate from, each with its own advantages and
    // drawbacks (speed, size, etc). We will first query info
    // about the available memory types: the physical device
    // memory properties function returns both memory heaps
    // (the distinct memory resources like VRAM and swap space
    // in RAM for when VRAM runs out), and the different types
    // of memory which exist within these heaps.
    let memory = instance.get_physical_device_memory_properties(data.physical_device);

    // We can now iterate over memory types to find one that is
    // suitable for the buffer:
    (0..memory.memory_type_count)
        .find(|&i| {
            // We are looking for a memory type that is both
            // suitable for the buffer and that has the
            // properties we want, which are specified in the
            // function parameters. The first condition is met
            // using the 'requirements' argument, containing a
            // "memory type bit field", where each bit
            // corresponds to a memory type and is set if it is
            // supported by the physical device (thus, a
            // suitable memory type is one whose corresponding
            // bit in the field is set to 1). The second
            // condition is that the properties of the memory
            // type match those required.
            requirements.memory_type_bits & (1 << i) != 0
                && memory.memory_types[i as usize].property_flags.contains(properties)
        })
        .ok_or(anyhow!(SuitabilityError("Failed to find suitable memory type.")))
}