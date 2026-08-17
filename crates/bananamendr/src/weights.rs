//! this_file: crates/bananamendr/src/weights.rs
//!
//! Loads `model.safetensors` into owned, shape-checked f32 tensors.
//!
//! The file is memory-mapped for reading and the mapping is dropped once the
//! tensors are materialised, so peak resident memory stays at roughly one copy
//! of the checkpoint. All published BananaMind-2 tensors are F32; anything else
//! is rejected rather than silently converted.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::error::Error;

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct Weights {
    tensors: HashMap<String, Tensor>,
}

impl Weights {
    pub fn load(path: &Path) -> Result<Self, Error> {
        let file = File::open(path).map_err(|e| Error::io(path, e))?;
        // SAFETY: the checkpoint is a read-only file we do not mutate; the
        // mapping is confined to this function.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| Error::io(path, e))?;
        let file = SafeTensors::deserialize(&mmap)?;

        let mut tensors = HashMap::new();
        for (name, view) in file.tensors() {
            if view.dtype() != Dtype::F32 {
                return Err(Error::Dtype {
                    name,
                    found: view.dtype(),
                });
            }
            let bytes = view.data();
            let mut data = Vec::with_capacity(bytes.len() / 4);
            data.extend(
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            );
            tensors.insert(
                name,
                Tensor {
                    shape: view.shape().to_vec(),
                    data,
                },
            );
        }
        Ok(Self { tensors })
    }

    /// Removes a tensor, checking its shape. Removal keeps peak memory at one
    /// copy while the model graph is being assembled.
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, Error> {
        let tensor = self
            .tensors
            .remove(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        if tensor.shape != shape {
            return Err(Error::Shape {
                name: name.to_string(),
                expected: shape.to_vec(),
                found: tensor.shape,
            });
        }
        Ok(tensor.data)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}
