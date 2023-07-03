use vulkanalia::prelude::v1_0::*;
use anyhow::{Result, anyhow};

pub unsafe fn create_shader_module(
    device: &Device,
    bytecode: &[u8],
) -> Result<vk::ShaderModule> {
    // Shader modules are a thin wrapper around the shader
    // bytecode loaded from a GLSL file. We included the
    // bytecode into our executable as an array of u8's, but the
    // Vulkan info struct builder expects u32's, so we need to
    // convert from one to the other. Storing first in a Vec
    // guarantess that the data is properly aligned, and it can
    // then be realigned to u32. The realignment method divides
    // the data into three parts, a suffix, a prefix, and a
    // middle section guaranteed to be properly aligned. If the
    // outside sections are not empty, it means bytecode data
    // was lost because it was not properly aligned in the first
    // place.
    let bytecode = Vec::<u8>::from(bytecode);
    let (prefix, code, suffix) = bytecode.align_to::<u32>();
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(anyhow!("Shader bytecode is not properly aligned."));
    }

    // The info struct takes in the bytecode slice size, and the
    // bytecode data itself.
    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.len())
        .code(code);

    // Then, the shader module can be created.
    Ok(device.create_shader_module(&info, None)?)
}