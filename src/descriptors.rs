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
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    // The second binding we want to set is for a combined
    // image sampler (that is, a descriptor type associated
    // with both an image resource and a sampler). This allows
    // combining the image and the sampler into a single
    // binding, which is more convenient than having to bind
    // them separately. This binding is for the fragment shader
    // stage (because that's where the color of the fragment is
    // going to be determined, although it is possible to use
    // it in the vertex shader stage, for example to
    // dynamically deform a grid of vertices by a heightmap)
    // and it is bound to the entry 1 in the shader.
    let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    // Then, the info struct and the actual layout may be
    // created.
    let bindings = [ubo_binding, sampler_binding];
    let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(&bindings);

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
    // sets are going to contain (UNIFORM_BUFFER, in this case)
    // and how many of them (one per swapchain image).
    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(data.swapchain_images.len() as u32);

    // Same for the combined image samplers, one per image.
    let sampler_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(data.swapchain_images.len() as u32);

    // Then, the pool can be createad, specifying its size and
    // the maximum number of sets that can be allocated from
    // it.
    let pool_sizes = &[ubo_size, sampler_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(data.swapchain_images.len() as u32);

    data.descriptor_pool = device.create_descriptor_pool(&info, None)?;

    info!("Descriptor pool created.");
    Ok(())
}

pub unsafe fn create_descriptor_sets(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Each descriptor set has a layout (the descriptor set
    // layout defined earlier) and a number of descriptors that
    // are bound to it (in our case, one per swapchain image).
    let layouts = vec![data.descriptor_set_layout; data.swapchain_images.len()];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts);

    data.descriptor_sets = device.allocate_descriptor_sets(&info)?;

    for i in 0..data.swapchain_images.len() {
        // The descriptor sets are allocated, but they are
        // empty. We need to specify the actual descriptors
        // that will be bound to them. In our case, we have a
        // first descriptor for each uniform buffer...
        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffers[i])
            .offset(0)
            .range(std::mem::size_of::<Mvp>() as u64);

        // ...and a second descriptor for the texture image,
        // which has an optimal layout for read-only shader
        // access and is configured to use the sampler we
        // created.
        let image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(data.texture_image_view)
            .sampler(data.texture_sampler);

        // Then, the descriptor set for the buffers can be
        // specified, with the following parameters: the
        // descriptor set to update (the i-th descriptor set in
        // the loop), the binding to update (0), the array
        // element to update (0, since we only have one element
        // per descriptor set), the descriptor type
        // (UNIFORM_BUFFER) and the buffer info for the
        // descriptors to update (there are also image_info for
        // image data and texel_buffer_view for buffer views
        // parameters, but we don't need them here).
        let buffer_infos = &[buffer_info];
        let buffer_set = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(buffer_infos)
            .build();
        
        // The same goes for the image descriptor set, with a
        // COMBINED_IMAGE_SAMPLER descriptor type, since it is
        // a texture combined with a sampler.
        let image_infos = &[image_info];
        let image_set = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(image_infos)
            .build();

        device.update_descriptor_sets(
            &[buffer_set, image_set], 
            &[] as &[vk::CopyDescriptorSet]
        );
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