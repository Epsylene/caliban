use crate::{devices::*, queues::QueueFamilyIndices};
use std::collections::HashSet;

use winit::window::Window;
use vulkanalia::{
    prelude::v1_0::*,
    window as vk_window,
    loader::{LibloadingLoader, LIBRARY},
    Version,
    vk::ExtDebugUtilsExtension,
};
use anyhow::{anyhow, Result};
use log::*;

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: vk::ExtensionName = vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

#[derive(Default)]
pub struct AppData {
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue: vk::Queue,
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
        
        // The next step involves choosing a physical device to
        // use on the system (the graphics card, for example),
        // and then creating a logical device to interface it
        // with the application.
        pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&entry, &instance, &mut data)?;

        Ok(Self { entry, instance, data, device })  
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        // Render the app here
        Ok(())
    }

    pub unsafe fn destroy(&mut self) {
        info!("Destroying the Vulkan instance.");
        
        self.device.destroy_device(None);

        if VALIDATION_ENABLED {
            self.instance.destroy_debug_utils_messenger_ext(self.data.debug_messenger, None);
        }

        self.instance.destroy_instance(None);
    }
}

unsafe fn create_instance(window: &Window, entry: &Entry, data: &mut AppData) -> Result<Instance> {
    info!("Creating the Vulkan instance.");
    
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

    Ok(instance)
}

unsafe fn create_logical_device(
    entry: &Entry, 
    instance: &Instance, 
    data: &mut AppData,
) -> Result<Device> {
    // The logical device serves as a layer between a physical
    // device and the application. There might be more than one
    // logical device per physical device, each representing
    // different sets of requirements. To create the logical
    // device, we need to build a representation of the queue
    // families of the physical device we are using, and in
    // particular to specify the number of queues for each queue
    // family; most drivers will only allow for a small number
    // of queues per family, but that is not a problem since the
    // command buffers can be created on multiple threads and
    // submitted all at once with minimal overhead. To build the
    // queue family info, we first need to get the indices of the
    // physical device queue families.
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    
    // We can then build the queues info struct. Vulkan lets you
    // give each queue a priority between 0.0 and 1.0, which
    // will influence the scheduling of command buffers
    // execution; we only have one queue, but we are still
    // required to provide the priorities, so we give it a value
    // of 1.
    let priorities = &[1.0];
    let queue = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(indices.graphics)
        .queue_priorities(priorities);

    // The next piece of information for the logical devices are
    // layers and extensions. Previous implementations of Vulkan
    // made a distinction between instance and device specific
    // validation layers, but this is no longer the case.
    // However, it is still a good idea to set them anyway to be
    // compatible with older implementations.
    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    // Some implementations are not fully conformant, so certain
    // Vulkan extensions need to be enabled to ensure
    // portability.
    // let mut extensions = Vec::new();
    // if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
    //     extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    // }

    // We can then specify the set of device features we are
    // going to use. Nothing special for now.
    let features = vk::PhysicalDeviceFeatures::builder();

    let queues = &[queue];
    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(queues)
        .enabled_layer_names(&layers)
        // .enabled_extension_names(&[extensions])
        .enabled_features(&features);

    // Finally, we can create the device, and set our handle to
    // the graphics queue.
    let device = instance.create_device(data.physical_device, &info, None)?;
    data.graphics_queue = device.get_device_queue(indices.graphics, 0);

    Ok(device)
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