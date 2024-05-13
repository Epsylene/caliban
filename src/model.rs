use std::{
    collections::HashMap,
    io::BufReader,
};

use anyhow::Result;
use glam::{vec2, vec3};

use crate::{app::AppData, vertex::Vertex};

pub fn load_model(path: &str, data: &mut AppData) -> Result<()> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);

    // Load the model from the file. The tobj crate loads both
    // models and materials, but we are only interested in
    // model data for now.
    let (models, _) = tobj::load_obj_buf(
        // Model data reader.
        &mut reader,
        // Use the default options for GPU loading:
        // triangulated faces (because that's what we work with
        // in our shader), single-indexed vertices (vertices
        // with the same position in space have the same
        // index), and discard degenerate faces (points or
        // lines not forming a triangle).
        &tobj::GPU_LOAD_OPTIONS,
        // Model loading function, which we don't care about.
        |_| Ok(Default::default()),
    )?;

    // There are a lot of vertices, but most are "repeated", in
    // the sense that they correspond to the same position in
    // space. Since the index buffer already stores the
    // correspondence between triangle vertices and 3D points,
    // we do not need to store every vertex of the OBJ in the
    // buffer, but only each unique one. This is done with a
    // hashmap.
    let mut unique = HashMap::new();

    // For each model...
    for model in &models {
        for index in &model.mesh.indices {
            let pos = &model.mesh.positions;
            let tex = &model.mesh.texcoords;
            
            // ...we can populate the vertex data from the
            // indices.
            let vertex = Vertex {
                pos: vec3(
                    pos[(3*index) as usize],
                    pos[(3*index + 1) as usize],
                    pos[(3*index + 2) as usize],
                ),
                color: vec3(1.0, 1.0, 1.0),
                // The texture coordinates are bottoñ-to-top in
                // the OBJ format (0 at the bottom), while ours
                // are top-to-bottom (0 at the top), so we need
                // to flip the vertical axis.
                texture: vec2(
                    tex[(2*index) as usize],
                    1.0 - tex[(2*index + 1) as usize],
                ),
            };

            // If the vertex is already in the list (that is,
            // if this position has already been visited), we
            // just need to push its index to the list;
            // otherwise, we add the vertex/index pair to the
            // map and to their corresponding buffers.
            if let Some(&index) = unique.get(&vertex) {
                data.indices.push(index as u32);
            } else {
                let index = data.vertices.len();
                unique.insert(vertex, index);
        
                data.vertices.push(vertex);
                data.indices.push(index as u32);
            }
        }
    }

    Ok(())
}