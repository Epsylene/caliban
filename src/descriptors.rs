use crate::app::AppData;

use vulkanalia::prelude::v1_0::*;
use anyhow::Result;
use glam::Mat4;
use log::*;

#[repr(C)]
struct Mvp {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
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