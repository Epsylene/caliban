use crate::devices::*;
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

#[derive(Default)]
pub struct AppData {
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub physical_device: vk::PhysicalDevice,
}

pub struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
}

impl App {
    pub unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        let instance = create_instance(window, &entry, &mut data)?;
        
        pick_physical_device(&instance, &mut data)?;

        Ok(Self { entry, instance, data })
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        // Render the app here
        Ok(())
    }

    pub unsafe fn destroy(&mut self) {
        info!("Destroying the Vulkan instance.");

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
    const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
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