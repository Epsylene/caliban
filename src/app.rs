use crate::{
    devices::*,
    swapchain::*, 
    pipeline::*,
    buffers::*,
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
    // - Render pass: the render pass describing the different
    //   framebuffer attachments and their usage
    // - Pipeline layout: the set of ressources that can be
    //   accessed by the pipeline
    // - Pipeline: the graphics pipeline, the succession of
    //   rendering stages in a single pass
    // - Framebuffer: collection of memory attachments used by a
    //   render pass instance
    // - Command buffers: buffers containing commands which can
    //   be submitted to a queue for execution
    // - Command pool: memory allocator for a set of command
    //   buffers
    // - Semaphores: synchronization primitive to insert a
    //   dependency within or accross command queue operations
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
    pub framebuffers: Vec<vk::Framebuffer>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub command_pool: vk::CommandPool,
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
}

pub struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    pub device: Device,
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
        // and then creating a logical device to interface with
        // the application.
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

        // The final step before actual rendering is to create
        // the framebuffers, which we use to bind the
        // attachments specified during render pass creation,
        // and the command buffers (allocated in a command
        // pool), to record them and submit them to the GPU.
        create_framebuffers(&device, &mut data)?;
        create_command_pool(&instance, &device, &mut data)?;
        create_command_buffers(&device, &mut data)?;

        // 
        create_sync_objects(&device, &mut data)?;

        Ok(Self { entry, instance, data, device })
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        // The first step in rendering The Triangle (TM) is to
        // acquire an image on the swapchain. The "acquire next
        // image" method takes in the swapchain from which to
        // acquire the image, a timeout value specifying how
        // long the function is to wait if no image is available
        // (in nanoseconds), a semaphore and/or a fence to
        // signal when the image is acquired, and returns the
        // index of the next available presentable image in the
        // swapchain.
        let image_index = self
            .device
            .acquire_next_image_khr(
                self.data.swapchain,
                u64::MAX,
                self.data.image_available_semaphore,
                vk::Fence::null()
            )?
            .0 as usize;

        // The queue submit operation is then configured with a
        // submit info struct, containing:
        //  - wait_semaphores: an array of semaphores upon which
        //    to wait before execution begins (here the "image
        //    available" semaphore);
        //  - wait_dst_stage_mask: an array of pipeline stages
        //    at which each corresponding semaphore wait will
        //    occur (here the color attachment output stage,
        //    after blending, when the final color values are
        //    output from the pipeline, since we want to wait
        //    writing colors to the image until it is truly
        //    available);
        //  - command_buffers: the command buffer handles to
        //    execute in the batch (here the command buffer at
        //    the index of the acquired image);
        //  - signal_semaphores: array of semaphores that will
        //    be signaled when the command buffers have
        //    completed execution (the "render finished"
        //    semaphore).
        let image_available = &[self.data.image_available_semaphore];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let render_finished = &[self.data.render_finished_semaphore];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(image_available)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(render_finished);

        // Lastly, the queue and its info can be submitted, with
        // an optional fence that will be signaled when the
        // workload is much larger.
        self.device.queue_submit(self.data.graphics_queue, &[submit_info], vk::Fence::null())?;

        // The final step of drawing a frame is submitting the
        // result back to the swapchain to have it eventually
        // show up on the screen. Presentation is configured
        // with an info struct detailing:
        //  - wait_semaphores: the semaphores on which to wait
        //    before presentation can happen, which are the
        //    "render finished" semaphores from before;
        //  - swapchains: the swapchains to present images to;
        //  - image_indices: the index of the image to present
        //    for each swapchain.
        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(render_finished)
            .swapchains(swapchains)
            .image_indices(image_indices);

        self.device.queue_present_khr(self.data.present_queue, &present_info)?;

        Ok(())
    }

    pub unsafe fn destroy(&mut self) {
        self.device.destroy_semaphore(self.data.render_finished_semaphore, None);
        self.device.destroy_semaphore(self.data.image_available_semaphore, None);
        self.device.destroy_command_pool(self.data.command_pool, None);

        self.data.framebuffers
            .iter()
            .for_each(|&f| self.device.destroy_framebuffer(f, None));

        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        
        self.data.swapchain_image_views
            .iter()
            .for_each(|&view| self.device.destroy_image_view(view, None));
        
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
    // around the idea of minimal driver overhead, there is very
    // little default error checking. Instead, Vulkan provides
    // "validation layers", which are optional components that
    // hook into Vulkan function calls to apply additional
    // checks and debug operations. Validation layers can only
    // be used if they have been installed onto the system, for
    // example as part of the LunarG Vulkan SDK. We first need
    // to get the list of available layers...
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

unsafe fn create_sync_objects(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Rendering operations, such as acquiring images,
    // presenting images or running a command buffer are
    // executed asynchronously. This means that the order of
    // execution is undefined, which poses a problem because
    // each operation depends on the completion of the previous
    // one. To solve this, Vulkan provides two ways of
    // synchronizing swapchain events: fences and semaphores.
    // Semaphores are simply signal identifiers that indicate
    // when a batch of commands has been processed. In our case,
    // we will need one semaphore to signal that an image has
    // been acquired and is ready for rendering, and one to
    // signal that rendering has finished and presentation can
    // happen.
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    data.image_available_semaphore = device.create_semaphore(&semaphore_info, None)?;
    data.render_finished_semaphore = device.create_semaphore(&semaphore_info, None)?;

    info!("Semaphores configured.");
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

    // If the callback returns true, the call is aborted with a
    // VALIDATION_FAILED error code; it should therefore only
    // return true when testing the validation layers
    // themselves.
    vk::FALSE
}