//! this_file: crates/bananamendr/src/matrix.rs
//!
//! A weight matrix in one of three forms: 32-bit floats, 8-bit codes, or ternary
//! codes.
//!
//! The engine reads the form from `config.json`, and the rest of the code does
//! not care which form arrived. A matrix keeps its codes in memory and rebuilds
//! the values inside the multiplication, so a small checkpoint stays small while
//! it runs.
//!
//! Both quantized forms use one scale for each group of columns. The ternary
//! form uses two scales: one for the positive weights and one for the negative
//! weights. `bananamendy quantize` writes these arrays, and
//! `bananamendy/src/bananamendy/quantize.py` documents how it chooses them.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::error::Error;

/// Below this many multiply-accumulates a quantized projection stays on the
/// calling thread. The value follows `ops::matvec`, which uses the same rule for
/// 32-bit floats.
#[cfg(feature = "parallel")]
const PARALLEL_MAC_THRESHOLD: usize = 1 << 18;

/// The name of the ternary method in `config.json`.
pub const TERNARY_METHODS: [&str; 2] = ["ternary-twn-v1", "ternary-gptq-v1"];
/// The name of the 8-bit method in `config.json`.
pub const INT8_METHOD: &str = "int8-sym-v1";

/// A weight matrix of shape `[rows, cols]`, row-major.
pub enum Matrix {
    /// 32-bit floats, as the original checkpoint holds them.
    Dense {
        rows: usize,
        cols: usize,
        data: Vec<f32>,
    },
    /// 8-bit codes. `value = code * scale[row, group]`.
    Int8 {
        rows: usize,
        cols: usize,
        group: usize,
        groups: usize,
        codes: Vec<i8>,
        scale: Vec<f32>,
    },
    /// Ternary codes, four in each byte, from the lowest bits upwards. The code
    /// 0 means minus one, 1 means zero, and 2 means plus one.
    Ternary {
        rows: usize,
        cols: usize,
        group: usize,
        groups: usize,
        codes: Vec<u8>,
        scale_positive: Vec<f32>,
        scale_negative: Vec<f32>,
    },
}

impl Matrix {
    pub fn rows(&self) -> usize {
        match self {
            Matrix::Dense { rows, .. }
            | Matrix::Int8 { rows, .. }
            | Matrix::Ternary { rows, .. } => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Matrix::Dense { cols, .. }
            | Matrix::Int8 { cols, .. }
            | Matrix::Ternary { cols, .. } => *cols,
        }
    }

    /// The name of the form, for a report to a user.
    pub fn kind(&self) -> &'static str {
        match self {
            Matrix::Dense { .. } => "float32",
            Matrix::Int8 { .. } => "int8",
            Matrix::Ternary { .. } => "ternary",
        }
    }

    /// The number of bytes that the matrix holds.
    pub fn bytes(&self) -> usize {
        match self {
            Matrix::Dense { data, .. } => data.len() * 4,
            Matrix::Int8 { codes, scale, .. } => codes.len() + scale.len() * 4,
            Matrix::Ternary {
                codes,
                scale_positive,
                scale_negative,
                ..
            } => codes.len() + (scale_positive.len() + scale_negative.len()) * 4,
        }
    }

    /// Builds an 8-bit matrix from the arrays of the checkpoint.
    pub fn int8(
        rows: usize,
        cols: usize,
        group: usize,
        codes: Vec<i8>,
        scale: Vec<f32>,
    ) -> Result<Self, Error> {
        let group = group.min(cols).max(1);
        let groups = cols.div_ceil(group);
        if codes.len() != rows * cols {
            return Err(Error::Quantization(format!(
                "8-bit codes hold {} values, and the shape needs {}",
                codes.len(),
                rows * cols
            )));
        }
        if scale.len() != rows * groups {
            return Err(Error::Quantization(format!(
                "8-bit scales hold {} values, and the shape needs {}",
                scale.len(),
                rows * groups
            )));
        }
        Ok(Matrix::Int8 {
            rows,
            cols,
            group,
            groups,
            codes,
            scale,
        })
    }

    /// Builds a ternary matrix from the arrays of the checkpoint.
    pub fn ternary(
        rows: usize,
        cols: usize,
        group: usize,
        codes: Vec<u8>,
        scale_positive: Vec<f32>,
        scale_negative: Vec<f32>,
    ) -> Result<Self, Error> {
        let group = group.min(cols).max(1);
        let groups = cols.div_ceil(group);
        let needed = (rows * cols).div_ceil(4);
        if codes.len() < needed {
            return Err(Error::Quantization(format!(
                "ternary codes hold {} bytes, and the shape needs {needed}",
                codes.len()
            )));
        }
        if scale_positive.len() != rows * groups || scale_negative.len() != rows * groups {
            return Err(Error::Quantization(format!(
                "ternary scales hold {} and {} values, and the shape needs {}",
                scale_positive.len(),
                scale_negative.len(),
                rows * groups
            )));
        }
        Ok(Matrix::Ternary {
            rows,
            cols,
            group,
            groups,
            codes,
            scale_positive,
            scale_negative,
        })
    }

    /// Reads one code of a ternary matrix. The result is -1, 0 or 1.
    #[inline]
    fn ternary_code(codes: &[u8], index: usize) -> i32 {
        let byte = codes[index >> 2];
        let shift = (index & 3) * 2;
        ((byte >> shift) & 0b11) as i32 - 1
    }

    /// Writes one row of the matrix into `out` as 32-bit floats.
    pub fn row_into(&self, row: usize, out: &mut [f32]) {
        match self {
            Matrix::Dense { cols, data, .. } => {
                out.copy_from_slice(&data[row * cols..(row + 1) * cols]);
            }
            Matrix::Int8 {
                cols,
                group,
                groups,
                codes,
                scale,
                ..
            } => {
                let row_codes = &codes[row * cols..(row + 1) * cols];
                let row_scale = &scale[row * groups..(row + 1) * groups];
                for (column, value) in out.iter_mut().enumerate() {
                    let index = (column / group).min(groups - 1);
                    *value = row_codes[column] as f32 * row_scale[index];
                }
            }
            Matrix::Ternary {
                cols,
                group,
                groups,
                codes,
                scale_positive,
                scale_negative,
                ..
            } => {
                let base = row * cols;
                let row_positive = &scale_positive[row * groups..(row + 1) * groups];
                let row_negative = &scale_negative[row * groups..(row + 1) * groups];
                for (column, value) in out.iter_mut().enumerate() {
                    let index = (column / group).min(groups - 1);
                    *value = match Self::ternary_code(codes, base + column) {
                        1 => row_positive[index],
                        -1 => -row_negative[index],
                        _ => 0.0,
                    };
                }
            }
        }
    }

    /// `out[i] = dot(row i, x)`.
    ///
    /// The rows are independent, so a large projection is spread over the Rayon
    /// pool, exactly as the projection of 32-bit floats is. Without that, a
    /// quantized checkpoint would run on one thread and lose more time than the
    /// smaller memory saves.
    pub fn matvec(&self, out: &mut [f32], x: &[f32]) {
        #[cfg(feature = "parallel")]
        {
            let macs = out.len() * self.cols();
            if !matches!(self, Matrix::Dense { .. }) && macs >= PARALLEL_MAC_THRESHOLD {
                let rows_per_task = ((1 << 15) / self.cols().max(1)).max(1);
                out.par_chunks_mut(rows_per_task)
                    .enumerate()
                    .for_each(|(chunk, rows)| {
                        let first = chunk * rows_per_task;
                        for (offset, value) in rows.iter_mut().enumerate() {
                            *value = self.row_dot(first + offset, x);
                        }
                    });
                return;
            }
        }
        self.matvec_serial(out, x)
    }

    /// The dot product of one row with `x`.
    pub fn row_dot(&self, row: usize, x: &[f32]) -> f32 {
        match self {
            Matrix::Dense { cols, data, .. } => {
                crate::ops::dot(&data[row * cols..(row + 1) * cols], x)
            }
            Matrix::Int8 {
                cols,
                group,
                groups,
                codes,
                scale,
                ..
            } => {
                let row_codes = &codes[row * cols..(row + 1) * cols];
                let row_scale = &scale[row * groups..(row + 1) * groups];
                let mut total = 0.0f32;
                for (index, chunk) in row_codes.chunks(*group).enumerate() {
                    let offset = index * group;
                    let mut inner = 0.0f32;
                    for (i, &code) in chunk.iter().enumerate() {
                        inner += code as f32 * x[offset + i];
                    }
                    total += inner * row_scale[index.min(groups - 1)];
                }
                total
            }
            Matrix::Ternary {
                cols,
                group,
                groups,
                codes,
                scale_positive,
                scale_negative,
                ..
            } => {
                let base = row * cols;
                let row_positive = &scale_positive[row * groups..(row + 1) * groups];
                let row_negative = &scale_negative[row * groups..(row + 1) * groups];
                let mut total = 0.0f32;
                for index in 0..*groups {
                    let start = index * group;
                    let end = (start + group).min(*cols);
                    let mut positive = 0.0f32;
                    let mut negative = 0.0f32;
                    for (offset, &value) in x[start..end].iter().enumerate() {
                        match Self::ternary_code(codes, base + start + offset) {
                            1 => positive += value,
                            -1 => negative += value,
                            _ => {}
                        }
                    }
                    total += positive * row_positive[index] - negative * row_negative[index];
                }
                total
            }
        }
    }

    fn matvec_serial(&self, out: &mut [f32], x: &[f32]) {
        match self {
            Matrix::Dense { data, .. } => crate::ops::matvec(out, data, x),
            _ => {
                for (row, value) in out.iter_mut().enumerate() {
                    *value = self.row_dot(row, x);
                }
            }
        }
    }

    /// Gives the dense values of the matrix. This copies, so it is for a caller
    /// that needs a plain matrix, for example a test.
    pub fn to_dense(&self) -> Vec<f32> {
        match self {
            Matrix::Dense { data, .. } => data.clone(),
            _ => {
                let (rows, cols) = (self.rows(), self.cols());
                let mut out = vec![0.0f32; rows * cols];
                for row in 0..rows {
                    let (start, end) = (row * cols, (row + 1) * cols);
                    self.row_into(row, &mut out[start..end]);
                }
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(codes: &[u8]) -> Vec<u8> {
        let mut packed = vec![0u8; codes.len().div_ceil(4)];
        for (index, &code) in codes.iter().enumerate() {
            packed[index >> 2] |= code << ((index & 3) * 2);
        }
        packed
    }

    #[test]
    fn ternary_rebuilds_the_values_that_the_codes_name() {
        // Two rows, four columns, one group: [-a, 0, a, a] and [a, a, 0, -a].
        let codes = pack(&[0, 1, 2, 2, 2, 2, 1, 0]);
        let matrix = Matrix::ternary(2, 4, 4, codes, vec![2.0, 3.0], vec![5.0, 7.0]).unwrap();
        assert_eq!(
            matrix.to_dense(),
            vec![-5.0, 0.0, 2.0, 2.0, 3.0, 3.0, 0.0, -7.0]
        );
    }

    #[test]
    fn ternary_matvec_equals_the_dense_multiplication() {
        let codes = pack(&[0, 1, 2, 2, 2, 2, 1, 0]);
        let matrix = Matrix::ternary(
            2,
            4,
            2,
            codes,
            vec![2.0, 3.0, 1.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let x = [1.0, 2.0, 3.0, 4.0];
        let dense = matrix.to_dense();
        let mut expected = vec![0.0f32; 2];
        crate::ops::matvec(&mut expected, &dense, &x);
        let mut actual = vec![0.0f32; 2];
        matrix.matvec(&mut actual, &x);
        for (a, b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-6, "{a} against {b}");
        }
    }

    #[test]
    fn int8_matvec_equals_the_dense_multiplication() {
        let codes: Vec<i8> = vec![127, -128, 0, 64, -32, 16, 8, -4];
        let matrix = Matrix::int8(2, 4, 2, codes, vec![0.5, 0.25, 1.0, 2.0]).unwrap();
        let x = [1.0, -2.0, 0.5, 3.0];
        let dense = matrix.to_dense();
        let mut expected = vec![0.0f32; 2];
        crate::ops::matvec(&mut expected, &dense, &x);
        let mut actual = vec![0.0f32; 2];
        matrix.matvec(&mut actual, &x);
        for (a, b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-4, "{a} against {b}");
        }
    }

    #[test]
    fn a_wrong_number_of_codes_is_an_error() {
        assert!(Matrix::int8(2, 4, 2, vec![1, 2, 3], vec![1.0, 1.0, 1.0, 1.0]).is_err());
        assert!(Matrix::ternary(2, 4, 2, vec![0], vec![1.0; 4], vec![1.0; 4]).is_err());
    }

    #[test]
    fn a_group_larger_than_the_row_becomes_one_group() {
        let matrix = Matrix::int8(1, 3, 64, vec![1, 2, 3], vec![2.0]).unwrap();
        assert_eq!(matrix.to_dense(), vec![2.0, 4.0, 6.0]);
        assert_eq!(matrix.kind(), "int8");
    }

    #[test]
    fn the_size_report_counts_the_codes_and_the_scales() {
        let matrix = Matrix::int8(2, 4, 2, vec![0; 8], vec![1.0; 4]).unwrap();
        assert_eq!(matrix.bytes(), 8 + 16);
        let dense = Matrix::Dense {
            rows: 2,
            cols: 4,
            data: vec![0.0; 8],
        };
        assert_eq!(dense.bytes(), 32);
    }
}
