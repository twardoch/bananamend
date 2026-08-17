//! this_file: crates/bananamendr/src/weights.rs
//!
//! Reads `model.safetensors` into owned, shape-checked f32 tensors.
//!
//! `load` opens a file and maps it into memory. The mapping goes away when the
//! tensors are complete, so the memory in use stays at approximately one copy of
//! the checkpoint. `from_bytes` does the same work on bytes that you already
//! have; a browser uses it, because a browser has no files.
//!
//! All published BananaMind-2 tensors are F32. The reader refuses a different
//! type; it does not convert it.

use std::collections::HashMap;

use safetensors::{Dtype, SafeTensors};

#[cfg(feature = "std-fs")]
use std::path::Path;

use crate::error::Error;

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct Weights {
    tensors: HashMap<String, Tensor>,
}

impl Weights {
    /// Reads a checkpoint file. The file is mapped into memory for reading.
    #[cfg(feature = "std-fs")]
    pub fn load(path: &Path) -> Result<Self, Error> {
        let file = std::fs::File::open(path).map_err(|e| Error::io(path, e))?;
        // SAFETY: the checkpoint is a read-only file that we do not change, and
        // the mapping does not leave this function.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Error::io(path, e))?;
        Self::from_bytes(&mmap)
    }

    /// Reads a checkpoint from bytes that you already have.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let file = SafeTensors::deserialize(bytes)?;

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
