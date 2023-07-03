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
    //  1) Input assembler: the vertices are collected into a
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

    // We will start by including the shader bytecode, compiled
    // from GLSL to SPIR-V with the compiler provided by the
    // Vulkan SDK, directly into the engine executable:
    let vert = include_bytes!("../shaders/shader.vert.spv");
    let frag = include_bytes!("../shaders/shader.frag.spv");

    // To use the shader, we have to create a "shader module", a
    // wrapper object around the shader bytecode.
    let vert_module = create_shader_module(device, &vert[..])?;
    let frag_module = create_shader_module(device, &frag[..])?;

    // Then we can define the first of the pipeline stages, the
    // vertex shader stage. Other than the stage name and the
    // shader bytecode, we also need to specify the entry point
    // of the shader program to build the stage (incidentally,
    // this means that we could write the code for different
    // stages in the same file, using different entry point
    // names). There is another, optional member,
    // 'specialization_info', which allows specifying shader
    // constants at build time, rather than at render time.
    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_module)
        .name(b"main\0");

    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_module)
        .name(b"main\0");

    device.destroy_shader_module(vert_module, None);
    device.destroy_shader_module(frag_module, None);

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