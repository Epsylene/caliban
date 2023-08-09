
use crate::{
    app::AppData,
    devices::SuitabilityError,
};

use glam::{Vec2, Vec3};
use vulkanalia::{
    vk::HasBuilder, 
    prelude::v1_0::*,
};
use anyhow::{anyhow, Result};
use log::info;
use lazy_static::lazy_static;
use std::ptr::copy_nonoverlapping as memcpy;

lazy_static! {
    static ref VERTICES: Vec<Vertex> = vec![
        Vertex::new(Vec2::new(0.0, -0.5), Vec3::new(1.0, 0.0, 0.0)),
        Vertex::new(Vec2::new(0.5, 0.5), Vec3::new(0.0, 1.0, 0.0)),
        Vertex::new(Vec2::new(-0.5, 0.5), Vec3::new(0.0, 0.0, 1.0)),
    ];
}

#[repr(C)]
pub struct Vertex {
    pos: Vec2,
    color: Vec3,
}

impl Vertex {
    pub fn new(pos: Vec2, color: Vec3) -> Self {
        Self { pos, color }
    }

    pub fn binding_description() -> vk::VertexInputBindingDescription {
        // After uploading the vertex data to the GPU, we need
        // to tell Vulkan how to pass it to the shader. The
        // first struct needed to convey this information is the
        // vertex binding info, used to describe the rate at
        // which to load data from memory throughout the
        // vertices, precising :
        //  - the binding index, an index into the array of
        //    buffers bound with vkCmdBindVertexBuffers;
        //  - the stride, the number of bytes between
        //    consecutive elements in the buffer;
        //  - the input rate, specifying whether to move to the
        //    next data entry after each vertex or after each
        //    instance.
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        // The second struct(s) is the vertex attribute
        // description. Each attribute description struct
        // describes how to extract a vertex attribute from a
        // chunk of vertex data originating from a binding
        // description. It contains:
        //  - the binding index, which is the binding number
        //    this attribute takes its data from;
        //  - the location index, which is the index of the
        //    current attribute in the array of attributes (0
        //    for position, 1 for color, for example), and the
        //    shader input location number for this attribute
        //    (the x in 'layout(location = x)')
        //  - the format of the attribute data, precising its
        //    size and type (a 2D position is a vec2 of signed
        //    floats, for example, so a R32G32_SFLOAT format)
        //  - the byte offset of the first element of the
        //    attribute relative to the beginning of the vertex
        //    data.
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0)
            .build();

        // The color attribute is very much the same as the
        // position attribute, except that it has a location of
        // 1, a R32G32B32_SFLOAT format (3 32-bit floats), and
        // an offset the size of the position attribute.
        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(std::mem::size_of::<Vec2>() as u32)
            .build();

        [pos, color]
    }
}

pub unsafe fn create_vertex_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Buffers in Vulkan are regions of memory used for storing
    // arbitrary data that can be read by the graphics card.
    // Vertex buffers are just buffers that contain vertex data;
    // to create them, we first need to fill an info struct
    // precising:
    //  - the size of the buffer, in bytes (which is the number
    //    of vertices multiplied by the size of a vertex in
    //    bytes);
    //  - the usage of the buffer, that is, whether it is a
    //    vertex buffer, index buffer, uniform buffer, etc;
    //  - the sharing mode, indicating the way the buffer will
    //    be used; either EXCLUSIVE, meaning that it will only
    //    be accessed by queue families that own it, or
    //    CONCURRENT, where it can be accessed by a number of
    //    specified queue families.
    let buffer_info = vk::BufferCreateInfo::builder()
        .size((std::mem::size_of::<Vertex>() * VERTICES.len()) as u64)
        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .build();

    data.vertex_buffer = device.create_buffer(&buffer_info, None)?;

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
    let requirements = device.get_buffer_memory_requirements(data.vertex_buffer);

    // Now that we have the requirements for the buffer memory,
    // we can actually build the memory allocation info struct,
    // with the size of the allocation and the index of the
    // memory type to use. The property flags of the memory type
    // are set to HOST_VISIBLE (the memory is allocated on the
    // host memory and accessible to the CPU) and HOST_COHERENT
    // (memory writes are visible both from the CPU and the GPU;
    // this is not trivial because memory writes are tipically
    // not done directly on memory, but on a cache first, which
    // might not be visible by all devices).
    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            requirements,
        )?)
        .build();

    // We can then actually allocate memory and bind it to the
    // vertex buffer if the allocation was successful, giving
    // the offset of the buffer in the allocated memory.
    data.vertex_buffer_memory = device.allocate_memory(&memory_info, None)?;
    device.bind_buffer_memory(data.vertex_buffer, data.vertex_buffer_memory, 0)?;

    // To copy the vertex data to the buffer, we first need to
    // map the buffer memory into CPU accessible memory (that
    // is, to obtain a CPU pointer into device memory), by
    // providing a memory ressource to access (the vertex buffer
    // memory) defined by an offset (0) and size (the size of
    // the buffer; it is also possible to specify the special
    // value WHOLE_SIZE to map all of memory) and some flags
    // (though there aren't any available yet in the current
    // API).
    let memory = device.map_memory(
        data.vertex_buffer_memory,
        0,
        buffer_info.size,
        vk::MemoryMapFlags::empty(),
    )?;

    // We can then copy the vertex data into the buffer memory
    // and then unmap it. We chose host coherence to deal with
    // the fact that the memory might not be changed directly
    // (for example because of caching); the other way to deal
    // with this problem is to manually flush the memory from
    // cache to memory after writing, and invalidate caches
    // before reading to force them to fetch the latest data
    // from VRAM. Host coherence may lead to slightly worse
    // performance than explicit flushing, but it is also
    // simpler. 
    memcpy(VERTICES.as_ptr(), memory.cast(), VERTICES.len());
    device.unmap_memory(data.vertex_buffer_memory);

    info!("Vertex buffer created.");
    Ok(())
}

unsafe fn get_memory_type_index(
    instance: &Instance,
    data: &mut AppData,
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