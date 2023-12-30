use crate::{
    app::AppData, 
    buffers::create_buffer
};

use vulkanalia::prelude::v1_0::*;
use anyhow::Result;
use glam::Mat4;
use log::*;

#[repr(C)]
pub struct Mvp {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

pub unsafe fn create_descriptor_set_layout(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // In Vulkan, a "descriptor" is an opaque data structure
    // representing a shader resource such as a buffer, image
    // view, sampler, etc. A descriptor set is a collection of
    // descriptors bound for the drawing commands; the types of
    // the resources that are going to be accessed by the
    // pipeline are specified with the descriptor set layout.
    // In the case of a uniform buffer, the descriptor set
    // contains a single descriptor of type UNIFORM_BUFFER,
    // accessed during the vertex shader stage, and bound to
    // the entry 0 in the shader.
    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .build();

    // Then, the info struct and the actual layout may be
    // created.
    let bindings = [ubo_binding];
    let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(&bindings)
        .build();

    data.descriptor_set_layout = device.create_descriptor_set_layout(&create_info, None)?;

    info!("Descriptor set layout created.");
    Ok(())
}

pub unsafe fn create_descriptor_pool(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // The descriptor pool is an allocation pool for descriptor
    // sets, just like the command pool is for command buffers.
    // We first need to describe the types of descriptors our
    // sets are going to contain (UNIFORM_BUFFER, in our case)
    // and how many of them (one per swapchain image).
    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(data.swapchain_images.len() as u32)
        .build();

    // Then, the pool can be createad, specifying its size and
    // the maximum number of sets that can be allocated from
    // it.
    let pool_sizes = &[ubo_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(data.swapchain_images.len() as u32)
        .build();

    data.descriptor_pool = device.create_descriptor_pool(&info, None)?;

    info!("Descriptor pool created.");
    Ok(())
}

pub unsafe fn create_descriptor_sets(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Each descriptor set has a layout (the descriptor set
    // layout defined earlier) and there is one for each image
    // in the swapchain.
    let layouts = vec![data.descriptor_set_layout; data.swapchain_images.len()];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts)
        .build();

    data.descriptor_sets = device.allocate_descriptor_sets(&info)?;

    for i in 0..data.swapchain_images.len() {
        // The descriptor sets are allocated, but they are
        // empty. We need to specify the actual descriptors
        // that will be bound to them. In our case, we have a
        // single descriptor of type UNIFORM_BUFFER, which is
        // bound to the entry 0 in the shader.
        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffers[i])
            .offset(0)
            .range(std::mem::size_of::<Mvp>() as u64)
            .build();

        // Then, the parameters for writing the descriptor set
        // are specified: the descriptor set to update (the
        // i-th descriptor set in the loop), the binding to
        // update (0), the array element to update (0, since we
        // only have one element per descriptor set), the
        // descriptor type (UNIFORM_BUFFER) and the buffer info
        // for the descriptors to update (there are also
        // image_info for image data and texel_buffer_view for
        // buffer views parameters, but we don't need them
        // here).
        let buffer_infos = &[buffer_info];
        let descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(buffer_infos)
            .build();

        device.update_descriptor_sets(&[descriptor_write], &[] as &[vk::CopyDescriptorSet]);
    }

    info!("Descriptor sets created.");
    Ok(())
}

pub unsafe fn create_uniform_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    data.uniform_buffers.clear();
    data.uniform_buffers_memory.clear();

    // A uniform buffer is a buffer that is made accessible in
    // a read-only fashion to shaders, so that they can read
    // constant parameter data. They are passed along through a
    // descriptor set, which is bound to the pipeline, so the
    // UBOs are the same for every shader (hence their name).
    // We should have multiple buffers, because multiple frames
    // may be in "in flight" (rendering) at the same time and
    // we don't want to update the buffer in preparation of the
    // next frame while a previous one is still reading from
    // it. Since we need to refer to the uniform buffer from
    // the command buffer, of which we have one per swapchain
    // image, it makes more sense to have one uniform buffer
    // per swapchain image too.
    for _ in 0..data.swapchain_images.len() {
        let (ubo, ubo_memory) = create_buffer(
            instance,
            device,
            data,
            std::mem::size_of::<Mvp>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER, 
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        )?;

        data.uniform_buffers.push(ubo);
        data.uniform_buffers_memory.push(ubo_memory);
    }
    
    info!("Uniform buffers created.");
    Ok(())
}