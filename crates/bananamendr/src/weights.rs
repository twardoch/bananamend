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

use crate::config::{Quantization, TensorQuantization};
use crate::matrix::{INT8_METHOD, Matrix, TERNARY_METHODS};

#[cfg(feature = "std-fs")]
use std::path::Path;

use crate::error::Error;

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// The codes of a quantized tensor, as bytes. The interpretation comes from the
/// `quantization` block of `config.json`.
pub struct RawTensor {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub bytes: Vec<u8>,
}

pub struct Weights {
    tensors: HashMap<String, Tensor>,
    raw: HashMap<String, RawTensor>,
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
        let mut raw = HashMap::new();
        for (name, view) in file.tensors() {
            // A quantized tensor arrives as bytes, and the engine keeps it that
            // way. Every other tensor must be F32.
            if view.dtype() == Dtype::U8 || view.dtype() == Dtype::I8 {
                raw.insert(
                    name,
                    RawTensor {
                        shape: view.shape().to_vec(),
                        dtype: view.dtype(),
                        bytes: view.data().to_vec(),
                    },
                );
                continue;
            }
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
        Ok(Self { tensors, raw })
    }

    /// Removes a tensor, checking its shape. Removal keeps peak memory at one
    /// copy while the model graph is being assembled.
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, Error> {
        let tensor = self
            .tensors
            .remove(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        if !shape.is_empty() && tensor.shape != shape {
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

    /// Takes a matrix in whichever form the checkpoint holds it.
    ///
    /// `quantization` comes from `config.json`. A name that it does not mention
    /// must be present as a tensor of 32-bit floats.
    pub fn take_matrix(
        &mut self,
        name: &str,
        shape: &[usize],
        quantization: Option<&Quantization>,
    ) -> Result<Matrix, Error> {
        let info = quantization.and_then(|q| q.tensors.get(name));
        match info {
            None => {
                let data = self.take(name, shape)?;
                Ok(Matrix::Dense {
                    rows: shape[0],
                    cols: shape[1],
                    data,
                })
            }
            Some(info) => self.take_quantized(name, shape, info, quantization),
        }
    }

    fn take_quantized(
        &mut self,
        name: &str,
        shape: &[usize],
        info: &TensorQuantization,
        quantization: Option<&Quantization>,
    ) -> Result<Matrix, Error> {
        if !info.shape.is_empty() && info.shape != shape {
            return Err(Error::Quantization(format!(
                "{name}: the block says shape {:?}, and the model needs {shape:?}",
                info.shape
            )));
        }
        let group = if info.group_size > 0 {
            info.group_size
        } else {
            quantization.map(|q| q.group_size).unwrap_or(0)
        };
        let (rows, cols) = (shape[0], shape[1]);

        if info.method == INT8_METHOD {
            let codes = self.take_raw(&format!("{name}.int8.codes"))?;
            let scale = self.take(&format!("{name}.int8.scale"), &[])?;
            let signed = codes.bytes.iter().map(|&b| b as i8).collect();
            return Matrix::int8(rows, cols, group, signed, scale);
        }
        if TERNARY_METHODS.contains(&info.method.as_str()) {
            let codes = self.take_raw(&format!("{name}.ternary.codes"))?;
            let positive = self.take(&format!("{name}.ternary.scale_pos"), &[])?;
            let negative = self.take(&format!("{name}.ternary.scale_neg"), &[])?;
            return Matrix::ternary(rows, cols, group, codes.bytes, positive, negative);
        }
        Err(Error::Quantization(format!(
            "{name}: the method {:?} is not known to this version",
            info.method
        )))
    }

    fn take_raw(&mut self, name: &str) -> Result<RawTensor, Error> {
        self.raw
            .remove(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}
