use crate::{
    app::AppData,
    devices::SuitabilityError, 
    commands::{begin_single_command, end_single_command},
};

use vulkanalia::prelude::v1_0::*;
use anyhow::{Result, anyhow};

pub unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    // Buffers in Vulkan are regions of memory used for storing
    // arbitrary data that can be read by the graphics card.
    // They are defined by their size (in bytes), their usage
    // (as vertex buffers, index buffers, uniform buffers, etc)
    // and their sharing mode, that is, how they will be
    // accessed: either only by queue families owning them
    // (EXCLUSIVE) or by a number of (previously specified)
    // queue families (CONCURRENT).
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    
    let buffer = device.create_buffer(&buffer_info, None)?;
    
    // After creating the buffer, we need to allocate memory for
    // it. To do so, we first need to get the memory
    // requirements of the buffer, which will get us 3 fields:
    //  - the size of the required amount of memory, in bytes;
    //  - the memory alignment, that is, the offset in bytes
    //    where the buffer begins in the allocated region of
    //    memory (one might allocate enough memory to fit
    //    several buffers, thus the need to tell the offset of a
    //    given buffer);
    //  - the memory type bits, a bit field of the memory types
    //    that are suitable for the buffer.
    let requirements = device.get_buffer_memory_requirements(buffer);
    
    // Now that we have the requirements for the buffer memory,
    // we can actually build the memory allocation info struct,
    // with the size of the allocation and the index of the
    // memory type to use based on the device requirements and
    // memory properties we want.
    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(find_memory_type(
            instance,
            data,
            properties,
            requirements,
        )?);

    // We can then actually allocate memory and bind it to the
    // vertex buffer if the allocation was successful, while
    // specifying the offset of the buffer in the allocated
    // memory.
    let buffer_memory = device.allocate_memory(&memory_info, None)?;
    device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory))
}

pub unsafe fn copy_buffer(
    device: &Device,
    data: &AppData,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<()> {
    // Copying data between buffers, like all commands, is done
    // with a command buffer. We will then first allocate and
    // begin a temporary command buffer for the transfer
    // operation.
    let command_buffer = begin_single_command(device, data)?;

    // We can then actually copy the data from the source buffer
    // to the destination buffer. We can define one or several
    // regions for the copy, each consisting of a source buffer
    // offset, a destination buffer offset, and a size.
    let regions = vk::BufferCopy::builder().size(size);
    device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);
    
    // Then, the command buffer can be ended to be submitted
    // for execution.
    end_single_command(device, data, command_buffer)?;

    Ok(())
}

pub unsafe fn find_memory_type(
    instance: &Instance,
    data: &AppData,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Result<u32> {
    // Graphics cards can offer different types of memory to
    // allocate from, each with its own advantages and drawbacks
    // (speed, size, etc). We will first query info about the
    // available memory types: the physical device memory
    // properties function returns both memory heaps (the
    // distinct memory resources like VRAM and swap space in RAM
    // for when VRAM runs out), and the different types of
    // memory which exist within these heaps.
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