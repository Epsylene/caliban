use crate::{
    app::AppData,
    shaders::*,    
};

use vulkanalia::prelude::v1_0::*;
use anyhow::Result;
use log::*;

pub unsafe fn create_render_pass(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // During rendering, the framebuffer will access different
    // attachments, like the color buffer or the depth buffer.
    // The render pass object specifies how these render targets
    // are configured, how many of them there are, how their
    // contents should be handled, etc.
    //
    // We will start by configuring the color buffer attachment,
    // represented by one of the images from the swapchain.
    // Apart from the image format, which should be the same
    // than the swapchain, and the number of fragment samples
    // (see multisampling in the pipeline creation), we also
    // need to specify:
    // - The loading and storing of data in the attachment at
    //   the beginning and end of the render pass. The loading
    //   can be either LOAD (preserve the existing data), CLEAR
    //   (clear the values to a constant) or DONT_CARE (existing
    //   contents undefined, we don't care about them). The
    //   storing can be either STORE (rendered contents are
    //   stored in memory so that they can be read later) or
    //   DONT_CARE (contents of the framebuffer will be
    //   undefined after the render pass);
    // - Stencil load and store operations: we won't be using
    //   the stencil buffer, so we can set these to DONT_CARE;
    // - The initial and final layout of the attachment:
    //   textures and framebuffers are images with a certain
    //   pixel format, whose layout can be changed depending on
    //   the current rendering operation. Some common layouts
    //   are COLOR_ATTACHMENT_OPTIMAL (optimized for images used
    //   as color attachments, before actual rendering),
    //   PRESENT_SRC_KHR (images to be presented in the
    //   swapchain) and TRANSFER_DST_OPTIMAL (destination for a
    //   memory copy operation). The UNDEFINED layout means that
    //   we don't care about the previous layout of the image,
    //   which is the case for the initial layout. We want the
    //   image to be ready for presentation at the end of the
    //   render pass, so we set the final layout to
    //   PRESENT_SRC_KHR.
    let color_attachment = vk::AttachmentDescription::builder()
        .format(data.swapchain_format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    // Render passes consist of multiple subpasses, subsequent
    // rendering operations that depend on the contents of
    // framebuffers in previous passes (for example a sequence
    // of post-processing effects that are applied one after
    // another). Dividing a render pass into subpasses allows
    // Vulkan to reorder the operations and conserve memory
    // bandwith for possibly better performance. Each subpass
    // references one or more attachments, through
    // AttachmentReference structs, which specify the index of
    // the attachment in the render pass (referenced in the
    // fragment shader with the "layout(location = 0)"
    // directive) and the layout of the attachment. Since our
    // render pass consists of a single subpass on a color
    // buffer attachment, the index is 0 and the layout is
    // COLOR_ATTACHMENT_OPTIMAL.
    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    
    // The subpass is explicitly stated to be a graphics subpass
    // (as opposed to a compute subpass, for example), and the
    // array of color attachments is passed to it. There can
    // also be input attachments (attachments read from a
    // shader), resolve attachments (used for multisampling
    // color attachments), depth stencil attachments (for depth
    // and stencil data) and preserve attachments (attachments
    // which are not used by the subpass, but must be preserved
    // for later use).
    let color_attachments = &[color_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments);

    // Subpass dependencies specify memory and execution
    // dependencies between subpasses. Although we have only a
    // single subpass now, the operations before and after this
    // subpass also count as implicit subpasses. The subpass
    // dependency struct is then built from:
    //  - A source subpass with index SUBPASS_EXTERNAL (implicit
    //    subpass) and a destination subpass with index 0;
    //  - A source and destination stages, both
    //    COLOR_ATTACHMENT_OUTPUT (final color values, after
    //    blending, since the image we want to present during
    //    our subpass is the final one in the pipeline)
    //  - A source and destination access mask. The source has
    //    no access flags, while the destination is marked as
    //    COLOR_ATTACHMENT_WRITE: these settings prevent the
    //    transition from happening until it's actually
    //    necessary (and allowed).
    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
    
    let attachments = &[color_attachment];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    // The render pass info struct can then finally be created,
    // containing both the attachments and the subpasses.
    let color_attachments = &[color_attachment];
    let subpasses = &[subpass];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(color_attachments)
        .subpasses(subpasses);

    data.render_pass = device.create_render_pass(&info, None)?;

    info!("Render pass created.");
    Ok(())
}

pub unsafe fn create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    // The graphics pipeline is the sequence of operations that
    // take the vertices and textures of the meshes all the way
    // to the pixels on the screen. It consists of the following
    // (simplified) stages:
    //
    //  1) Input assembly: the vertices are collected into a
    //     structure; repetition may be avoided with the use of
    //     index buffers.
    //  2) Vertex shader: each vertex is applied some
    //     transformation, tipically from model space to screen
    //     space, and per-vertex data is passed down the
    //     pipeline.
    //  3) Tesselation shader: geometry may be subdivided based
    //     on certain rules, to add detail to the mesh (for
    //     example when it is close to the camera).
    //  4) Geometry shader: each primitive (triangle, line,
    //     point) can be discarded or output even more
    //     primitives. This is more flexible than the
    //     tesselation shader, but not much used in practice
    //     because it is less performant on most modern cards.
    //  5) Rasterization: the primitives are discretized into
    //     fragments, an abstraction of a pixel. Any fragments
    //     that fall outside of the screen are discarded, and
    //     the vertex shader attributes are interpolated across
    //     the fragments. Fragments that are behind other
    //     primitive fragments are discarded as well in the
    //     early depth test stage, which takes place just before
    //     the fragment shader.
    //  6) Fragment shader: each fragment is written to a
    //     framebuffer with a color and depth values. This may
    //     be done using interpolated data from the vertex
    //     shader stage, like texture coordinates and surface
    //     normals.
    //  7) Color blending: fragments that map to the same pixel
    //     are mixed together as defined by the blending
    //     operation (overwrite, add, transparency mix, etc).
    //
    // The input assembler, rasterization and color blending
    // stages are known as "fixed-function" stages, because they
    // are not programmable by the user, only configured. The
    // other stages are all user-programmable, and may be
    // skipped if not needed.

    // Before the first stage, input assembly, we have to define
    // vertex input state -- the format of the vertex data that
    // will be passed to the vertex shader. There are two main
    // things to consider:
    //  - Bindings: the spacing between data and whether the
    //    data is per-vertex or per-instance;
    //  - Attributes: the type of the vertex attributes (color,
    //    position, normal, etc), which binding to load them
    //    from and at which offset.
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();

    // The input assembly info struct describes the kind of
    // geometry that will be drawn from the vertices and if
    // primitive restart should be enabled. Some of the possible
    // topology values are POINT_LIST (points from vertices),
    // LINE_LIST (disjoint lines from each pair of vertices,
    // without reuse), LINE_STRIP (continuous chain of lines
    // between the first and last indexed vertices),
    // TRIANGLE_LIST (disjoint triangle from every triplet of
    // vertices, without reuse), TRIANGLE_STRIP (triangle paving
    // of the surface of vertices) and TRIANGLE_FAN (triangle
    // paving centered around the first indexed vertex).
    // Enabling primitive restart allows to break up lines and
    // triangles in the LINE_STRIP and TRIANGLE_STRIP modes by
    // using a special index (either 0xFFFF or 0xFFFFFFFF
    // depending on the index size).
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    // Then comes the vertex shader stage. We will start by
    // including the shader bytecode, compiled from GLSL to
    // SPIR-V with the compiler provided by the Vulkan SDK,
    // directly into the engine executable, and create a "shader
    // module", a wrapper object passed to Vulkan and containing
    // the shader bytecode.
    let vert = include_bytes!("../shaders/shader.vert.spv");
    let vert_module = create_shader_module(device, &vert[..])?;

    // Other than the stage name and the shader bytecode, we
    // also need to specify the entry point of the shader
    // program to build the stage (incidentally, this means that
    // we could write the code for different stages in the same
    // file, using different entry point names). There is
    // another, optional member, 'specialization_info', which
    // allows specifying shader constants at build time, rather
    // than at render time.
    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_module)
        .name(b"main\0");

    // The next step involves configuring the viewport, which is
    // the region of the framebuffer that the output will be
    // rendered to. It is defined by a rectangle from (x, y) to
    // (x + width, y + height). Furthermore, the range of depth
    // values to use for the framebuffer can be specified with
    // min and max values between 0 and 1.
    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    // The viewport defines the transformation from the image to
    // the framebuffer, but the actual pixel region to store in
    // the framebuffer is defined by the scissor rectangle (for
    // example, one could define a viewport surface on the whole
    // window, but a scissor rectangle on half of the image,
    // such that the other half is rendered as white pixels).
    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D::default())
        .extent(data.swapchain_extent);

    // The viewport and scissor rectangle are then combined into
    // a viewport state struct, which is passed to the pipeline.
    let viewports = &[viewport];
    let scissors = &[scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(viewports)
        .scissors(scissors);

    // The next stage, the rasterizer, takes the geometry shaped
    // by the vertex shader and turns it into fragments to be
    // colored by the fragment shader, while performing some
    // tests like depth testing (comparing the value of a
    // generated fragment with the value stored in the depth
    // buffer, and replacing it if the new value is closer),
    // face culling (test if the fragment is in a front- or
    // back-facing polygon) and the scissors test (discarding
    // fragments outside of the scissor rectangle). Several
    // other options can be configured:
    //  - Depth clamp: fragments beyond the near and far planes
    //    are clamped rather than discarded;
    //  - Rasterizer discard: all geometry is discarded before
    //    the rasterization stage, which effectively disables
    //    it. This can be useful when doing non-graphical or
    //    pre-rendering tasks that need the work from the vertex
    //    and tesselation stages.
    //  - Polygon mode: how fragments are generated for
    //    geometry, either in FILL (fill the area of the
    //    polygon), LINE (draw the edges of the polygon) or
    //    POINT (draw the polygon vertices) mode.
    //  - Line width: the thickness of the line in terms of
    //    number of fragments.
    //  - Cull mode: which faces to cull, either FRONT, BACK,
    //    NONE or FRONT_AND_BACK.
    //  - Front face: how to determine the front-facing faces,
    //    either the CLOCKWISE or COUNTER_CLOCKWISE oriented
    //    ones.
    //  - Depth bias: the depth value of a fragment can be
    //    offset by some value (the bias), to avoid z-fighting
    //    when polygons are coplanar. This offset can be
    //    constant or based on the slope of the polygon (which
    //    can be useful when using it to reduce shadow acne, the
    //    erroneous shadows appearing on a model when applying a
    //    shadow map because the faces self-intersect; faces are
    //    more likely to self-shadow if they are facing the
    //    light, which is why slope-scaled depth bias is
    //    useful).
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(false);

    // Multisampling can then be configured to perform
    // antialiasing. In ordinary rendering, the pixel color is
    // based on a single sample point; if the polygon edge
    // passes through the pixel but is not close enough to the
    // sample point, the pixel will be left blank, leading to a
    // jagged "staircase" effect. Multisample antialiasing
    // (MSAA), however, takes multiple samples per pixel and
    // averages them to determine the final pixel color,
    // producing a smoother result. Sample shading, which
    // applies the multisampling to every fragment in the
    // polygon (not only at the edges), may be used too; this
    // can be useful when there is a low-res texture with high
    // contrasting colors, that won't be antialised with normal
    // MSAA. We will not be using antialiasing for now, so we
    // will disable sample shading and set the number of samples
    // to 1.
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::_1);

    // After rasterization comes the fragment shader. As with
    // the vertex shader, we will include the shader bytecode
    // directly into the executable, create a shader module, and
    // set up the fragment stage.
    let frag = include_bytes!("../shaders/shader.frag.spv");
    let frag_module = create_shader_module(device, &frag[..])?;
    
    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_module)
        .name(b"main\0");

    // The final stage is color blending. Since there is already
    // a color in the framebuffer (from the previous frame), we
    // need to choose how to combine it with the new color
    // returned by the fragment shader. This is what we call
    // color blending, which is performed in several steps:
    //  1) Multiply the "source" (new) and "destination" (old)
    //     colors by a factor, differentiating between the RGB
    //     and alpha components. This is called the "blend
    //     factor" and is set to a value between 0 and 1 (giving
    //     1 to the source color and 0 to the destination color
    //     replaces the old color with the new one, for
    //     example);
    //  2) Mixing the source and destination colors with a
    //     blending operation, which can be ADD (add the two
    //     colors), SUBTRACT (subtract the source from the
    //     destination), REVERSE_SUBTRACT, MIN/MAX (take the
    //     minimum/maximum of the two colors), etc. The RGB and
    //     alpha components can again be mixed separately.
    //  3) Applying a mask (the "color write mask") to the final
    //     color, which can be used to disable writing to the
    //     framebuffer for certain components (for example, if
    //     we only want to write to the red and blue channels).
    // The most common way to use color blending is to implement
    // alpha blending, which is used to simulate transparency by
    // mixing the source and destination colors based on its
    // opacity, with a linear blend alpha*new + (1 - alpha)*old.
    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(false);
    
    // The color blend attachment struct is then attached to the
    // color blend struct; there can be several color blend
    // attachements, one for each framebuffer. The color blend
    // state struct allows a second method of blending by
    // performing a bitwise combination of the color components
    // with the provided operator (COPY, AND, OR, CLEAR, etc);
    // this will automatically disable the first method. It also
    // allows to set the "blend constants", a RGBA vector used
    // in some blend factors (like BLEND_FACTOR_CONSTANT_COLOR
    // or BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA).
    let attachments = &[attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .attachments(attachments);

    // The ressources that can be accessed by the pipeline, like
    // uniforms (global data shared across shaders) or push
    // constants (small bunches of data sent to a shader), are
    // described with a pipeline layout object; ours will be
    // empty for now.
    let layout_info = vk::PipelineLayoutCreateInfo::builder();
    data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    // We can now combine all of the structures and objects
    // above to create the actual graphics pipeline. It is
    // comprised of shader stages (the shader modules that
    // define the functionality of the programmable stages of
    // the pipeline), fixed-function state (the structures that
    // define the fixed-function stages of the pipeline),
    // pipeline layout (the uniform and push values referenced
    // by the shader that can be updated at runtime) and render
    // pass (the attachments referenced by the pipeline stages
    // and their usage). We finally pass the index of the
    // subpass where this pipeline will be used.
    //
    // There are two more parameters left, the "base pipeline
    // handle" and "base pipeline index". These are used when
    // deriving a pipeline from an existing one, which can be
    // less expensive when they have a lot of functionality in
    // common; switching between pipelines from the same parent
    // is also faster. We won't be referencing another pipeline
    // for now, so we specify a null handle and an invalid
    // index.
    let stages = &[vert_stage, frag_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&color_blend_state)
        .layout(data.pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(0)
        .base_pipeline_handle(vk::Pipeline::null())
        .base_pipeline_index(-1);

    // The pipeline creation function takes an array of pipeline
    // info structs and creates multiple pipeline objects in a
    // single call. The first parameter, the pipeline cache, is
    // used to store and reuse the results of pipeline creation
    // calls, which can speed up the whole process.
    data.pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
        .0[0];

    device.destroy_shader_module(vert_module, None);
    device.destroy_shader_module(frag_module, None);

    info!("Pipeline created.");
    Ok(())
}