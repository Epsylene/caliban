use winit::window::Window;
use vulkanalia::{
    prelude::v1_0::*,
    window as vk_window,
    loader::{LibloadingLoader, LIBRARY},
    Version,
};
use anyhow::{anyhow, Result};
use log::*;

pub struct App {
    entry: Entry,
    instance: Instance,
}

impl App {
    pub unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let instance = Self::create_instance(window, &entry)?;
        
        Ok(Self { entry, instance })
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        // Render the app here
        Ok(())
    }

    pub unsafe fn destroy(&mut self) {
        info!("Destroying the Vulkan instance.");
        self.instance.destroy_instance(None);
    }

    unsafe fn create_instance(window: &Window, entry: &Entry) -> Result<Instance> {
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
        // window integration and convert them to C strings
        let mut extensions = vk_window::get_required_instance_extensions(window)
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();

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
        // extensions info
        let info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_extension_names(&extensions)
            .flags(flags);

        // We can give a custom allocator to the instance, but
        // here we set it to None
        entry.create_instance(&info, None).map_err(Into::into)
    }
}