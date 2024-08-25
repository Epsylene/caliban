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

use log::info;
use caliban::{
    core::queues::get_graphics_family_index,
    renderer::VALIDATION_LAYER,
};

fn main() {
    std::env::set_var("RUST_LOG", "info");
    pretty_env_logger::init();
    
    // Vulkan entry point
    let entry = unsafe {
        let loader = LibloadingLoader::new(LIBRARY).unwrap();
        Entry::new(loader).unwrap()
    };

    // Application info and validation layers
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"buffer-alloc\0")
        .application_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 3, 0));

    let layers = [VALIDATION_LAYER.as_ptr()];

    let info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers);

    // Vulkan instance
    let instance = unsafe { entry.create_instance(&info, None).unwrap() };

    // Physical device
    let (physical_device, graphics_queue) = unsafe {
        instance
            .enumerate_physical_devices()
            .unwrap()
            .iter()
            .find_map(|&physical_device| {
                if let Ok(queue_index) = get_graphics_family_index(&instance, physical_device) {
                    Some((physical_device, queue_index))
                } else {
                    None
                }
            })
            .unwrap()
    };

    // Logical device
    let priorities = &[1.0];
    let graphics_queues = &[
        vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(graphics_queue)
            .queue_priorities(priorities)
    ];

    let create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(graphics_queues)
        .enabled_layer_names(&layers);

    let device = unsafe { instance.create_device(physical_device, &create_info, None).unwrap() };
    info!("Created device.");

    todo!("allocate a buffer");
}