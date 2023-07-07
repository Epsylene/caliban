use crate::{
    devices::*,
    shaders::*,
    swapchain::*,
};
use std::collections::HashSet;

use winit::window::Window;
use vulkanalia::{
    prelude::v1_0::*,
    window as vk_window,
    loader::{LibloadingLoader, LIBRARY},
    Version,
    vk::{ExtDebugUtilsExtension, AmigoProfilingSubmitInfoSEC},
    vk::KhrSurfaceExtension,
    vk::KhrSwapchainExtension,
};
use anyhow::{anyhow, Result};
use log::*;

pub const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
pub const VALIDATION_LAYER: vk::ExtensionName = vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
pub const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

#[derive(Default)]
pub struct AppData {
    // - Surface: the Vulkan surface object, an abstraction of
    //   the native window object on which to render images
    // - Debug messenger: a messenger object for handling
    //   callbacks from the validation layers (that is, to print
    //   messages from the validation layers with our log system
    //   instead of the standard output)
    // - Physical device: the physical device to use for the
    //   graphics, in other words the graphics card (most of the
    //   time)
    // - Graphics queue: where graphics commands are sent to
    //   while waiting for execution
    // - Presentation queue: queue for rendering images to the
    //   surface
    // - Swapchain: an abstraction for an array of presentable
    //   images associated with a surface
    // - Swapchain images: the images controlled by the
    //   swapchain
    // - Swapchain format: the format of the swapchain images
    // - Swapchain extent: the resolution of the swapchain
    //   images
    // - Swapchain image views: views of the swapchain images,
    //   which describe how they should be accessed
    // - Render pass: the render pass describing the different
    //   framebuffer attachments and their usage
    // - Pipeline layout: the set of ressources that can be
    //   accessed by the pipeline
    // - Pipeline: the graphics pipeline, the succession of
    //   rendering stages in a single pass
    pub surface: vk::SurfaceKHR,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub render_pass: vk::RenderPass,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

pub struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
}

impl App {
    pub unsafe fn create(window: &Window) -> Result<Self> {
        // To create a Vulkan instance, we first need a special
        // function loader to load the initial commands from the
        // Vulkan DLL. Next we create an entry point using this
        // loader, and finally use the entry point, window
        // handle and application data to create the Vulkan
        // instance.
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        let instance = create_instance(window, &entry, &mut data)?;
        
        // Since Vulkan is a platform agnostic API, it does not
        // interface directly with the window system on its own;
        // instead, it exposes surface objects, abstract
        // representations of native window objects to render
        // images on. As with any other Vulkan object, the
        // creation of the surface involves first filling an
        // info struct (that takes the Win32 window handles on
        // Windows, for example) and then actually creating the
        // object; however, Vulkanalia provides a convenient
        // function to handle the platform differences for us
        // and return a proper Vulkan surface.
        data.surface = vk_window::create_surface(&instance, &window, &window)?;
        info!("Surface created.");

        // The next step involves choosing a physical device to
        // use on the system (the graphics card, for example),
        // and then creating a logical device to interface it
        // with the application.
        pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&entry, &instance, &mut data)?;

        // We then have to create the swapchain, which is the
        // structure presenting rendered images to the surface,
        // and the swapchain image views, which are the actual
        // way Vulkan accesses the swapchain images.
        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;

        // Then enter the graphics pipeline, the succession of
        // rendering stages that go from taking a bunch of
        // vertices to presenting a properly shaded set of
        // pixels to the screen. Before that, however, we need
        // to tell Vulkan about the framebuffer attachments that
        // will be used while rendering: the object containing
        // this information is called a "render pass".
        create_render_pass(&instance, &device, &mut data)?;
        create_pipeline(&device, &mut data)?;

        Ok(Self { entry, instance, data, device })
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        // Render the app here
        Ok(())
    }

    pub unsafe fn destroy(&mut self) {
        self.data.swapchain_image_views.iter().for_each(|&view| {
            self.device.destroy_image_view(view, None);
        });

        self.device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
        self.device.destroy_device(None);
        
        self.instance.destroy_surface_khr(self.data.surface, None);
        
        if VALIDATION_ENABLED {
            self.instance.destroy_debug_utils_messenger_ext(self.data.debug_messenger, None);
        }
        
        self.instance.destroy_instance(None);
        info!("Destroyed the Vulkan instance.");
    }
}

unsafe fn create_instance(window: &Window, entry: &Entry, data: &mut AppData) -> Result<Instance> {
    // Validation layers: because the Vulkan API is designed
    // around the idea of minimal driver overhead, there is
    // very little default error checking. Instead, Vulkan
    // provides "validation layers", which are optional
    // components that hook into Vulkan function calls to
    // apply additional checks and debug operations.
    // Validation layers can only be used if they have been
    // installed onto the system, for example as part of the
    // LunarG Vulkan SDK. We then first need to get the list
    // of available layers...
    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    // ...then check if validation layers are available...
    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer not available."));
    }
    
    // ...and finally put in our layers list, which we will
    // give to Vulkan later.
    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    // Application info: application name and version,
    // engine name and version, and Vulkan API version. The
    // Vulkan API version is required and must be set to
    // 1.0.0 or greater.
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"caliban-app\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"caliban\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    // Extensions: enumerate the required extensions for
    // window integration and convert them to C strings.
    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    // If the validation layers are enabled, we add the
    // debut utils extension to set up a callback for the
    // validation layer messages.
    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    // Some platforms have not a fully compliant Vulkan
    // implementation, and need since v1.3.216 of the Vulkan
    // API to enable special portability extensions. One of
    // those platforms is none other than macOS, so we check
    // the target OS and the Vulkan API version to enable
    // those extensions if needed.
    let flags = if
        cfg!(target_os = "macos") &&
        entry.version()? >= PORTABILITY_MACOS_VERSION
    {
        info!("Enabling extensions for macOS portability.");
        extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name.as_ptr());
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    }
    else {
        vk::InstanceCreateFlags::empty()
    };

    // Instance info: combines the application and
    // extensions info, and enables the given layers
    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    // Debug info: set up a debug messenger for the
    // validation layers, that calls our debug callback
    // function to print messages for all severity levels
    // and types of events.
    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        // Vulkan structs, like the instance info, have the
        // ability to be extended with other structs,  
        info = info.push_next(&mut debug_info);
    }

    // We can give a custom allocator to the instance, but
    // we set it here to None.
    let instance = entry.create_instance(&info, None)?;

    if VALIDATION_ENABLED {
        // Create the debug messenger in the instance with
        // our debug info and link it to our app data
        data.debug_messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
    }

    info!("Vulkan instance created.");
    Ok(instance)
}

unsafe fn create_render_pass(
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

unsafe fn create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
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

    // A limited amount of the state previously specified can be
    // changed at runtime without recreating the pipeline, like
    // the viewport size, the line width or blend constants.
    // This will ignore the configuration of these values and
    // require to specify them at drawing time.
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
        .dynamic_states(&[
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
        ]);

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
        .dynamic_state(&dynamic_state)
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
    data.pipeline = device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?.0[0];

    device.destroy_shader_module(vert_module, None);
    device.destroy_shader_module(frag_module, None);

    info!("Pipeline created.");
    Ok(())
}

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut std::ffi::c_void,
) -> vk::Bool32 {
    // The debug callback function ensures that we print
    // messages with our own log system instead of the
    // standard output. The 'extern "system"' bit links the
    // function to the system ABI, instead of the Rust one,
    // which is required for Vulkan to use it directly;
    // furthermore, the function prototype needs to match
    // that of vk::PFN_vkDebugUtilsMessengerCallbackEXT,
    // which specifies four parameters:
    //  1) 'messageSeverity': the importance of the message,
    //     as standard DEBUG, WARNING, ERR, ..., log levels
    //  2) 'messageType': the type of event associated,
    //     either GENERAL (unrelated to the specification),
    //     VALIDATION (specification violation) or
    //     PERFORMANCE (non-optimal use of the API)
    //  3) 'pCallbackData': the debug message data
    //  4) 'pUserData': a pointer to user-defined data, here
    //     unused

    let data = unsafe { *data };
    let message = unsafe { std::ffi::CStr::from_ptr(data.message) }.to_string_lossy();
    
    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({type_:?}) {message}");
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({type_:?}) {message}");
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({type_:?}) {message}");
    } else {
        trace!("({type_:?}) {message}");
    }

    // If the callback returns true, the call is aborted
    // with a VALIDATION_FAILED error code; it should then
    // only return true when testing the validation layers
    // themselves.
    vk::FALSE
}