use crate::{
    commands::*, 
    devices::*, 
    frame::*, 
    image::*, 
    swapchain::*,
    sync::*,
};

use std::collections::HashSet;

use winit::window::Window;
use vulkanalia::{
    prelude::v1_0::*,
    vk::DeviceV1_3,
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
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Default)]
pub struct RenderData {
    pub surface: vk::SurfaceKHR,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue: vk::Queue,
    pub graphics_queue_family: u32,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_format: vk::Format,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_extent: vk::Extent2D,
    pub frames: [FrameData; MAX_FRAMES_IN_FLIGHT],
}

pub struct Renderer {
    // - Entry: the Vulkan entry point, the first function to
    //   call to load the Vulkan library
    // - Instance: the Vulkan instance, the handle to the Vulkan
    //   library and the first object to create
    // - Data: the application data, containing all the objects
    //   necessary for rendering
    // - Device: the logical device, the interface to the
    //   physical device and the object to create other Vulkan
    //   objects
    // - Window: handle to the OS window
    // - Frame: the current frame in the swapchain
    // - Resized: whether the window has been resized
    entry: Entry,
    instance: Instance,
    data: RenderData,
    pub device: Device,    
    frame: usize,
}

impl Renderer {
    pub unsafe fn create(window: &Window) -> Result<Self> {
        // To create a Vulkan instance, we first need a special
        // function loader to load the initial commands from
        // the Vulkan DLL. Next we create an entry point using
        // this loader, and finally use the entry point, window
        // handle and application data to create the Vulkan
        // instance.
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = RenderData::default();
        let instance = create_instance(window, &entry, &mut data)?;
        
        // Since Vulkan is a platform agnostic API, it does not
        // interface directly with the window system on its
        // own; instead, it exposes surface objects, abstract
        // representations of native window objects to render
        // images on. As with any other Vulkan object, the
        // creation of the surface involves first filling an
        // info struct (that takes the Win32 window handles on
        // Windows, for example) and then actually creating the
        // object; however, Vulkanalia provides a convenient
        // function to handle the platform differences for us
        // and return a proper Vulkan surface.
        data.surface = vk_window::create_surface(&instance, window, window)?;
        info!("Surface created.");

        // The next step involves choosing a physical device to
        // use on the system (the graphics card, for example),
        // and then creating a logical device to interface with
        // the application.
        data.physical_device = pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&entry, &instance, &mut data)?;

        // We then have to create the swapchain, which is the
        // structure presenting rendered images to the surface,
        // and the swapchain image views, which are the actual
        // way Vulkan accesses the swapchain images.
        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;

        // The final step before actual rendering is to:
        //  - Create the command pools, to allocate memory for
        // the command buffers;
        //  - Create the command buffers, to record the
        //    commands that will be executed on the GPU.
        create_command_pools(&instance, &device, &mut data)?;
        create_command_buffers(&device, &mut data)?;

        // Finally, we create the synchronization objects to
        // ensure that the CPU and GPU are in sync when
        // rendering.
        create_sync_objects(&device, &mut data)?;

        Ok(Self { 
            entry, 
            instance, 
            data, 
            device, 
            frame: 0,
        })
    }

    pub unsafe fn render(&mut self) -> Result<()> {
        // The first step is to acquire an image on the
        // swapchain. Before that, however, we need to wait for
        // the previous frame to finish rendering, which is
        // done by waiting for the fence corresponding to that
        // frame. The wait_for_fences function also takes a
        // boolean value to wait either for all or any of the
        // fences to be signaled, and a timeout value to wait
        // for.
        let frame = &mut self.data.frames[self.frame];
        self.device.wait_for_fences(
            &[frame.in_flight_fence],
            true, 
            u64::MAX
        )?;

        // After completing, the fence is restored to the
        // unsignaled state for the coming frame.
        self.device.reset_fences(&[frame.in_flight_fence])?;
        
        // The "acquire next image" method takes in the
        // swapchain from which to acquire the image, a timeout
        // value specifying how long the function is to wait if
        // no image is available (in nanoseconds), a semaphore
        // and/or a fence to signal when the image is acquired,
        // and returns a result on the index of the next
        // available presentable image in the swapchain.
        let index_result = self.device
            .acquire_next_image_khr(
                self.data.swapchain,
                u64::MAX,
                frame.image_available_semaphore,
                vk::Fence::null()
            );
        
        // The result contains the index of the acquired image
        // in the swapchain, but if the swapchain is no longer
        // adequate for rendering (for example, if the window
        // has been resized), the result is either an
        // OUT_OF_DATE error (the swapchain has become
        // incompatible with the surface and can no longer be
        // used for rendering) or a SUBOPTIMAL error (the
        // swapchain can still be used, but the surface
        // properties are no longer matched exactly). In the
        // first case, we have to recreate the swapchain.
        let image_index = match index_result {
            Ok((index, _)) => index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                return Err(anyhow!("Swapchain out of date."));
            },
            Err(e) => return Err(anyhow!("Failed to acquire next image: {:?}", e)),
        };

        // Command buffers are allocated from pools and
        // recorded with commands to send to the GPU. Changing
        // commands dynamically requires changing the buffers,
        // one way or another. Vulkan offers 3 basic approaches
        // to this problem:
        //  1) Reset the command buffer, which clears the
        //     commands recorded to it, and record new ones.
        //     This is simple enough, but frequent calls to
        //     vkResetCmdBuffer might add an overhead.
        //  2) Free the command buffer and allocate a new one.
        //     This is much less efficient (up to 25 times
        //     slower!) than resetting buffers, because it
        //     involves repeated calls to the CPU for
        //     allocating and deallocating the memory.
        //  3) Reset the command pool, which resets all the
        //     buffers allocated from it in one go. This can be
        //     even more performant than method 1 (by a factor
        //     of 2).
        //
        // We will start by using the first method, since we
        // only have one buffer per frame.
        self.device.reset_command_buffer(
            frame.main_buffer, 
            vk::CommandBufferResetFlags::empty()
        )?;

        // The command buffer can then be started recording,
        // specifying usage with some parameters:
        //  - flags: either ONE_TIME_SUBMIT (the command buffer
        //    will be rerecorded right after executing it
        //    once), RENDER_PASS_CONTINUE (secondary command
        //    buffers that are entirely within a single render
        //    pass) and SIMULTANEOUS_USE (the command buffer
        //    can be resubmitted while it is in the pending
        //    state). We choose the first option because we
        //    will be recording the command buffer every frame;
        //  - inheritance info: only used for secondary command
        //    buffers, this specifies which state to inherit
        //    from the calling primary command buffer.
        let inheritance = vk::CommandBufferInheritanceInfo::builder();
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .inheritance_info(&inheritance);

        self.device.begin_command_buffer(frame.main_buffer, &info)?;

        // Then, we can start by transitioning the swapchain
        // image into a drawable layout, to clear the color.
        let image = self.data.swapchain_images[image_index];
        transition_image_layout(
            &self.device, 
            frame.main_buffer, 
            image,
            vk::ImageLayout::UNDEFINED, 
            vk::ImageLayout::GENERAL
        )?;

        // We will clear this image with a 120-frame flashing
        // blue color; the subresource range affected is the
        // color bit.
        let clear_color = vk::ClearColorValue {
            float32: [0.0, 0.0, 1.0, 1.0],
        };

        let ranges = &[subresource_range(vk::ImageAspectFlags::COLOR)];
        self.device.cmd_clear_color_image(
            frame.main_buffer, 
            image, 
            vk::ImageLayout::GENERAL,
            &clear_color, 
            ranges
        );

        // Now, the image can be transitioned again for
        // presentation to the surface.
        transition_image_layout(
            &self.device, 
            frame.main_buffer,
            image, 
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::PRESENT_SRC_KHR 
        )?;

        // All commands have been recorded, so the command
        // buffer can be ended.
        self.device.end_command_buffer(frame.main_buffer)?;

        // The next step is to prepare the submission for the
        // queue. There are two semaphores to signal, the
        // "image available" semaphore, which waits for
        // COLOR_ATTACHMENT_OUTPUT, the stage where final color
        // values are output from the pipeline...
        let wait_info = &[semaphore_submit(
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            frame.image_available_semaphore
        )];

        // ...and the "render finished" semaphore, which
        // signals the end of the execution of all graphics
        // pipeline stages.
        let signal_info = &[semaphore_submit(
            vk::PipelineStageFlags2::ALL_GRAPHICS,
            frame.render_finished_semaphore
        )];

        // Furthermore, we have submit info on the command
        // buffer that is to be executed.
        let cmd_info = &[vk::CommandBufferSubmitInfo::builder()
            .command_buffer(frame.main_buffer)];

        // We can then put these together and actually submit
        // the queue.
        let submit_info = vk::SubmitInfo2::builder()
            .wait_semaphore_infos(wait_info)
            .signal_semaphore_infos(signal_info)
            .command_buffer_infos(cmd_info);

        // The "in-flight fence" is set by the queue submit
        // operation so that when rendering of the next frame
        // is started on the CPU, it will wait for the GPU to
        // finish the previous frame before submitting
        // commands.
        self.device.queue_submit2(
            self.data.graphics_queue,
            &[submit_info],
            frame.in_flight_fence
        )?;

        // The final step is to present the image to the
        // surface. The present info struct takes the
        // semaphores to wait on and signal, the swapchain to
        // present to, and the index of the image to present.
        let wait_semaphores = &[frame.render_finished_semaphore];
        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(wait_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        // The present operation is then executed on the queue,
        // and the frame counter is incremented.
        self.device.queue_present_khr(self.data.graphics_queue, &present_info)?;
        
        self.frame += 1;
        self.frame %= MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    pub unsafe fn destroy(&mut self) {
        destroy_swapchain(&self.device, &self.data);

        self.data.frames
            .iter()
            .for_each(|f| self.device.destroy_command_pool(f.command_pool, None));

        destroy_sync_objects(&self.device, &mut self.data);

        self.instance.destroy_surface_khr(self.data.surface, None);
        self.device.destroy_device(None);

        if VALIDATION_ENABLED {
            self.instance.destroy_debug_utils_messenger_ext(self.data.debug_messenger, None);
        }
        
        self.instance.destroy_instance(None);
        info!("Destroyed the Vulkan instance.");
    }
}

unsafe fn create_instance(window: &Window, entry: &Entry, data: &mut RenderData) -> Result<Instance> {
    // Validation layers: because the Vulkan API is designed
    // around the idea of minimal driver overhead, there is
    // very little default error checking. Instead, Vulkan
    // provides "validation layers", which are optional
    // components that hook into Vulkan function calls to apply
    // additional checks and debug operations. Validation
    // layers can only be used if they have been installed onto
    // the system, for example as part of the LunarG Vulkan
    // SDK. We first need to get the list of available
    // layers...
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

    // Application info: application name and version, engine
    // name and version, and Vulkan API version. The Vulkan API
    // version is required and must be set to 1.0.0 or greater.
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"caliban-app\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"caliban\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 3, 0));

    // Extensions: enumerate the required extensions for window
    // integration and convert them to C strings.
    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    // If the validation layers are enabled, we add the debut
    // utils extension to set up a callback for the validation
    // layer messages.
    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    // Some platforms have not a fully compliant Vulkan
    // implementation, and need since v1.3.216 of the Vulkan
    // API to enable special portability extensions. One of
    // those platforms is none other than macOS, so we check
    // the target OS and the Vulkan API version to enable those
    // extensions if needed.
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

    // Instance info: combines the application and extensions
    // info, and enables the given layers
    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    // Debug info: set up a debug messenger for the validation
    // layers, that calls our debug callback function to print
    // messages for all severity levels and types of events.
    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        // Vulkan structs, like the instance info, have the
        // ability to be extended with other structs, which can
        // in turn be extended with other structs, and so on.
        // In this case, we are extending the instance info
        // with the debug info if the validation layers are
        // enabled, which will be used to create the debug
        // messenger.
        info = info.push_next(&mut debug_info);
    }

    // We can give a custom allocator to the instance, but we
    // set it here to None.
    let instance = entry.create_instance(&info, None)?;

    if VALIDATION_ENABLED {
        // Create the debug messenger in the instance with our
        // debug info and link it to our app data
        data.debug_messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
    }

    info!("Vulkan instance created.");
    Ok(instance)
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