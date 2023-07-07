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
    vk::ExtDebugUtilsExtension,
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
    // - Pipeline layout: the set of ressources that can be
    //   accessed by the pipeline
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
    pub pipeline_layout: vk::PipelineLayout,
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

        // Then enter the graphics pipeline (the succession of
        // shader stages that go from taking a bunch of vertices
        // to presenting a properly shaded set of pixels to the
        // screen):
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
    // whole window, but a scissor rectangle on half of the
    // image, such that the other half is rendered as white
    // pixels).
    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D::default())
        .extent(data.swapchain_extent);

    // The viewport and scissor rectangle are then combined into
    // a viewport state struct, which is passed to the pipeline.
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&[viewport])
        .scissors(&[scissor]);

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
    let frag = include_bytes!("../shaders/shader.vert.spv");
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
    // bitwise combination of the color components with the
    // provided operator (COPY, AND, OR, CLEAR, etc); this will
    // automatically disable the first method. It also allows to
    // set the "blend constants", a RGBA vector used in some
    // blend factors (like BLEND_FACTOR_CONSTANT_COLOR or
    // BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA).
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .attachments(&[attachment]);

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