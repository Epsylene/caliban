use crate::{
    commands::*, 
    depth::*, 
    descriptors::*, 
    devices::*, 
    image::create_color_objects, 
    model::load_model, 
    pipeline::*, 
    swapchain::*, 
    texture::*, 
    vertex::*
};

use std::{
    collections::HashSet,
    time::Instant,
    ptr::copy_nonoverlapping as memcpy,
};

use glam::{
    Mat4, 
    vec3
};

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
    // - Command pool: memory pool for a set of command buffers
    // - Semaphores: GPU-GPU synchronization primitive to insert
    //   a dependency within or accross command queue operations
    // - Fences: CPU-GPU synchronization primitive to insert a
    //   dependency between queues and the host
    // - Vertex buffer: buffer containing vertex data to upload
    //   to the GPU
    // - Device memory: opaque handle to device memory
    // - Index buffer: buffer containing the indices for each
    //   vertex in the vertex buffer
    // - Descriptor set layout: the layout of a descriptor set
    //   object, a collection of opaque structures representing
    //   a shader resource each
    // - Descriptor sets: the actual descriptors sets bound to
    //   the pipeline, one per swapchain image
    // - Descriptor pool: memory pool for the descriptor sets
    // - Uniform buffers: ressource buffers used for read-only
    //   global data in shaders
    // - Texture image: image used as a texture in the shaders
    // - Texture sampler: sampler object used to sample a
    //   texture
    // - Depth image: buffer used for depth testing
    // - Vertices/indices: vertex and index data for the
    //   current loaded model
    // - Mip levels: the number of mipmap levels in the texture
    // - MSAA samples: the number of samples to use for MSAA
    //   (antialisaing)
    // - Color image: render target for the MSAA operation
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
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub rendering_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub descriptor_pool: vk::DescriptorPool,
    pub uniform_buffers: Vec<vk::Buffer>,
    pub uniform_buffers_memory: Vec<vk::DeviceMemory>,
    pub texture_image: vk::Image,
    pub texture_image_memory: vk::DeviceMemory,
    pub texture_image_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,
    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub mip_levels: u32,
    pub msaa_samples: vk::SampleCountFlags,
    pub color_image: vk::Image,
    pub color_image_memory: vk::DeviceMemory,
    pub color_image_view: vk::ImageView,
}

pub struct App {
    // - Entry: the Vulkan entry point, the first function to
    //   call to load the Vulkan library
    // - Instance: the Vulkan instance, the handle to the Vulkan
    //   library and the first object to create
    // - Data: the application data, containing all the objects
    //   necessary for rendering
    // - Device: the logical device, the interface to the
    //   physical device and the object to create other Vulkan
    //   objects
    // - Frame: the current frame in the swapchain
    // - Resized: whether the window has been resized
    entry: Entry,
    instance: Instance,
    data: AppData,
    pub device: Device,
    frame: usize,
    pub resized: bool,
    start: Instant,
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
        // to tell Vulkan about the framebuffer attachments
        // that will be used while rendering: the object
        // containing this information is called a "render
        // pass". The descriptor set layout, specifying the
        // types of ressources accessed by the pipeline in
        // order to be used in the shaders, is also created
        // here.
        create_render_pass(&instance, &device, &mut data)?;
        create_descriptor_set_layout(&device, &mut data)?;
        create_pipeline(&device, &mut data)?;

        // The final step before actual rendering is to:
        //  - Create the command pool, to allocate memory for
        // the command buffers; 
        //  - The color objects (buffer for the MSAA operation)
        //  - The depth objects (depth buffer and related
        // objects), to provide depth information to the scene; 
        //  - The texture image and its view; 
        //  - The framebuffers, which we use to bind the
        // attachments specified during render pass creation;
        //  - Load the OBJ model to render; 
        //  - Allocate the vertex buffers and index buffers, to
        // later populate vertex data for the GPU; 
        //  - The uniform buffers to send ressources to the
        // shaders; 
        //  - The descriptor pool to allocate the descriptor
        // sets these ressources are bound to; 
        //  - The actual descriptors sets;
        //  - The command buffers (allocated in a command
        // pool), to record them and submit them to the GPU.
        create_command_pool(&instance, &device, &mut data)?;
        create_color_objects(&instance, &device, &mut data)?;
        create_depth_objects(&instance, &device, &mut data)?;
        create_framebuffers(&device, &mut data)?;
        create_texture_image("res/viking_room.png", &instance, &device, &mut data)?;
        create_texture_image_view(&device, &mut data)?;
        create_texture_sampler(&device, &mut data)?;
        load_model("res/viking_room.obj", &mut data)?;
        create_vertex_buffer(&instance, &device, &mut data)?;
        create_index_buffer(&instance, &device, &mut data)?;
        create_uniform_buffer(&instance, &device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_descriptor_sets(&device, &mut data)?;
        create_command_buffers(&device, &mut data)?;

        // To handle CPU-GPU and GPU-GPU synchronization, we
        // have to create several sync objects like fences and
        // semaphores.
        create_sync_objects(&device, &mut data)?;

        Ok(Self { entry, instance, data, device, frame: 0, resized: false, start: Instant::now() })
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        // The first step in rendering The Triangle (TM) is to
        // acquire an image on the swapchain. Before that,
        // however, we need to wait for the previous frame to
        // finish rendering, which is done by waiting for the
        // fence corresponding to that frame.
        self.device.wait_for_fences(
            &[self.data.rendering_fences[self.frame]], 
            true, 
            u64::MAX
        )?;

        // The "acquire next image" method takes in the
        // swapchain from which to acquire the image, a timeout
        // value specifying how long the function is to wait if
        // no image is available (in nanoseconds), a semaphore
        // and/or a fence to signal when the image is acquired,
        // and returns a result on the index of the next
        // available presentable image in the swapchain.
        let result = self
            .device
            .acquire_next_image_khr(
                self.data.swapchain,
                u64::MAX,
                self.data.image_available_semaphores[self.frame],
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
        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!("Failed to acquire next image: {}", e)),
        };

        // If the image is already in flight, we wait for the
        // fence corresponding to the acquired image to be
        // signaled (in other words, we wait for it to be
        // available).
        if !self.data.images_in_flight[image_index].is_null() {
            self.device.wait_for_fences(
                &[self.data.images_in_flight[image_index]], 
                true, 
                u64::MAX
            )?;
        }

        // Once it has been signaled, the image is in render by
        // the current frame, so we set the "image in flight"
        // fence to the "rendering" fence for this frame.
        self.data.images_in_flight[image_index] = self.data.rendering_fences[self.frame];

        // When the image is signaled as available and set to
        // the "render" state, we can also update the uniform
        // buffer, since it means that the previous image has
        // completed rendering.
        self.update_uniform_buffer(image_index)?;

        // We can now configure the queue submit operation with
        // a submit info struct, containing:
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
        let image_available = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let render_finished = &[self.data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(image_available)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(render_finished);

        // The rendering fences are then restored to their
        // unsignaled state.
        self.device.reset_fences(&[self.data.rendering_fences[self.frame]])?;

        // Lastly, the queue and its info can be submitted, as
        // well as the fence corresponding to the current frame.
        self.device.queue_submit(
            self.data.graphics_queue, 
            &[submit_info], 
            self.data.rendering_fences[self.frame]
        )?;

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

        // Just like when we acquired the image, presenting
        // images returns a result on the optimality of the
        // swapchain.
        let result = self.device.queue_present_khr(self.data.present_queue, &present_info);
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

        // Then, if the swapchain is suboptimal or out-of-date,
        // we recreate it. It is important to do this after
        // presentation to ensure that the semaphores are in a
        // consistent state, otherwise a signalled semaphore may
        // never be properly waited upon. We also check on the
        // flag 'resized' in case the platform does not trigger
        // an OUT_OF_DATE error when the window is resized.
        if changed || self.resized {
            self.resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(e) = result {
            return Err(anyhow!("Failed to present queue: {}", e));
        }
        
        // The current frame in the swapchain is increased by 1
        // within the MAX_FRAMES_IN_FLIGHT limit, which is the
        // maximum number of frames processed simultaneously.
        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        debug!("Rendering...");
        Ok(())
    }

    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        // The uniform buffer we are sending is comprised of a
        // view matrix looking from above at (2,2,2)...
        let view = Mat4::look_at_rh(
            vec3(2.0, 2.0, 2.0), 
            vec3(0.0, 0.0, 0.0), 
            vec3(0.0, 0.0, 1.0));

        // ...and a perspective projection matrix for a 45º
        // FOV, an aspect ratio matching that of the window,
        // and a near and far plane at 0.1 and 10.0.
        let mut proj = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4, 
            self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32, 
            0.1, 
            10.0
        );

        proj.y_axis.y *= -1.0;

        // Those can then be combined in the MVP object...
        let vp = Vp { view, proj };

        // ...and mapped to the uniform buffer memory (that is,
        // the uniform buffer in the device memory) for the
        // current image in the swapchain...
        let memory = self.device.map_memory(
            self.data.uniform_buffers_memory[image_index],
            0, 
            std::mem::size_of::<Vp>() as u64, 
            vk::MemoryMapFlags::empty()
        )?;

        // ...and finally copied.
        memcpy(&vp, memory.cast(), 1);
        self.device.unmap_memory(self.data.uniform_buffers_memory[image_index]);
        
        debug!("Updated uniform buffer.");
        Ok(())
    }

    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        // Even with a fully functional swapchain, it is
        // possible for the window surface to change such that
        // the swapchain is no longer compatible with it (after
        // a window resize, for example). The swapchain and all
        // the objects that depend on it or the window must then
        // be recreated. We will first call a function to wait
        // on the device until all ressources are free to use,
        // and destroy the swapchain before recreating it.
        self.device.device_wait_idle()?;
        self.destroy_swapchain();
        
        // Then, the swapchain is recreated. The images views
        // are recreated because they depend on the swapchain
        // images; the render pass is recreated because it
        // depends on the format of the swapchain images; the
        // pipeline is recreated because it defines viewport
        // and scissor rectangle sizes (note that it is
        // possible to define these as dynamic state to avoid
        // redefining them); depth objects, framebuffers and
        // uniform buffers can then be recreated, and finally
        // descriptors and command buffers.
        create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
        create_swapchain_image_views(&self.device, &mut self.data)?;
        create_render_pass(&self.instance, &self.device, &mut self.data)?;
        create_pipeline(&self.device, &mut self.data)?;
        create_color_objects(&self.instance, &self.device, &mut self.data)?;
        create_depth_objects(&self.instance, &self.device, &mut self.data)?;
        create_framebuffers(&self.device, &mut self.data)?;
        create_uniform_buffer(&self.instance, &self.device, &mut self.data)?;
        create_descriptor_pool(&self.device, &mut self.data)?;
        create_descriptor_sets(&self.device, &mut self.data)?;
        create_command_buffers(&self.device, &mut self.data)?;
        
        // Lastly, we resize our list of fences for the new
        // swapchain, since there is a possibility that there
        // might be a different number of swapchain images after
        // recreation.
        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());

        Ok(())
    }

    pub unsafe fn destroy(&mut self) {
        self.destroy_swapchain();

        self.device.destroy_image(self.data.texture_image, None);
        self.device.destroy_image_view(self.data.texture_image_view, None);
        self.device.destroy_sampler(self.data.texture_sampler, None);
        self.device.free_memory(self.data.texture_image_memory, None);
        self.device.destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
        self.device.destroy_buffer(self.data.index_buffer, None);
        self.device.free_memory(self.data.index_buffer_memory, None);
        self.device.destroy_buffer(self.data.vertex_buffer, None);
        self.device.free_memory(self.data.vertex_buffer_memory, None);

        self.data.rendering_fences
            .iter()
            .for_each(|&f| self.device.destroy_fence(f, None));
        
        self.data.render_finished_semaphores
            .iter()
            .for_each(|&s| self.device.destroy_semaphore(s, None));
        self.data.image_available_semaphores
            .iter()
            .for_each(|&s| self.device.destroy_semaphore(s, None));
        
        self.device.destroy_command_pool(self.data.command_pool, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance.destroy_debug_utils_messenger_ext(self.data.debug_messenger, None);
        }
        
        self.instance.destroy_instance(None);
        info!("Destroyed the Vulkan instance.");
    }

    unsafe fn destroy_swapchain(&mut self) {
        // Color image
        self.device.destroy_image_view(self.data.color_image_view, None);
        self.device.destroy_image(self.data.color_image, None);
        self.device.free_memory(self.data.color_image_memory, None);
        
        // Depth image
        self.device.destroy_image_view(self.data.depth_image_view, None);
        self.device.destroy_image(self.data.depth_image, None);
        self.device.free_memory(self.data.depth_image_memory, None);
        
        // Descriptor pool
        self.device.destroy_descriptor_pool(self.data.descriptor_pool, None);

        // Uniform buffers
        self.data.uniform_buffers
            .iter()
            .for_each(|&b| self.device.destroy_buffer(b, None));

        self.data.uniform_buffers_memory
            .iter()
            .for_each(|&m| self.device.free_memory(m, None));

        // Framebuffers
        self.data.framebuffers
            .iter()
            .for_each(|&f| self.device.destroy_framebuffer(f, None));

        // Command buffers, pipeline, render pass
        self.device.free_command_buffers(self.data.command_pool, &self.data.command_buffers);
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        
        // Swapchain image views
        self.data.swapchain_image_views
            .iter()
            .for_each(|&view| self.device.destroy_image_view(view, None));
        
        // Swapchain
        self.device.destroy_swapchain_khr(self.data.swapchain, None);

        info!("Destroyed the swapchain and related objects.");
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
        // ability to be extended with other structs, which can
        // in turn be extended with other structs, and so on. In
        // this case, we are extending the instance info with
        // the debug info if the validation layers are enabled,
        // which will be used to create the debug messenger.
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
    // synchronizing swapchain events: fences and semaphores. We
    // have to take care of setting the SIGNALED flag when
    // creating the fences, because they are in the unsignaled
    // state by default, which will freeze the program when the
    // render function waits for the fences to be signaled the
    // first time.
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder()
        .flags(vk::FenceCreateFlags::SIGNALED);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        // Semaphores are simply signal identifiers that
        // indicate when a batch of commands has been processed.
        // In our case, we will need one semaphore to signal
        // that an image has been acquired and is ready for
        // rendering, and one to signal that rendering has
        // finished and presentation can happen.
        data.image_available_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.render_finished_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);

        // Fences are similar to semaphores, but they have
        // accessible state and can be waited for from the
        // program code; thus, they can insert a dependency
        // between a queue and the host, which means that they
        // are used for CPU-GPU synchronization, while
        // semaphores handle GPU-GPU synchronization. For
        // example, if the CPU is submitting work faster than
        // the GPU can process it, semaphores and command
        // buffers will be used for multiple frames at the same
        // time: creating a fence for each frame in the
        // swapchain will allow us to wait for objects to finish
        // executing while having multiple frames "in-flight"
        // (worked on asynchronously).
        data.rendering_fences.push(device.create_fence(&fence_info, None)?);
    }

    // In-flight fences avoid concurrent usage of command
    // buffers and semaphores due to high CPU frequencies, but
    // if images are returned by the swapchain out-of-order then
    // it's possible that we may start rendering to a swapchain
    // image that is already "in flight". To avoid this, we need
    // to track for each image in the swapchain if a frame in
    // flight is currently using it, by refering to the
    // corresponding fence. Because no frame uses an image
    // initially, we explicitly initialize each image fence to
    // null.
    data.images_in_flight = data.swapchain_images
        .iter()
        .map(|_| vk::Fence::null())
        .collect();

    info!("Sync objects configured.");
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