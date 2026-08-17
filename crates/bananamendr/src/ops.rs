//! this_file: crates/bananamendr/src/ops.rs
//!
//! Scalar f32 kernels. Everything is plain safe Rust so the same code compiles
//! and behaves identically on macOS (aarch64/x86_64) and Windows; autovectorisation
//! by LLVM is relied on instead of intrinsics.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Projections below this many multiply-accumulates are computed on the calling
/// thread; above it, blocks of rows are spread across the Rayon pool. Small
/// projections lose more to scheduling than they gain from parallelism. Both
/// constants were picked by sweeping them on an M4 Max against all three
/// checkpoints (`bananamendr bench`); see WORK.md for the measurements.
const PARALLEL_MAC_THRESHOLD: usize = 1 << 18;

/// Minimum multiply-accumulates per Rayon task. One row of a 640-wide
/// projection is far too little work to pay for a task, so rows are batched.
#[cfg(feature = "parallel")]
const MACS_PER_TASK: usize = 1 << 15;

/// `out[i] = dot(w[i, ..], x)` for a row-major `w` of shape `[out_dim, in_dim]`.
pub fn matvec(out: &mut [f32], w: &[f32], x: &[f32]) {
    let in_dim = x.len();
    debug_assert_eq!(w.len(), out.len() * in_dim);

    // WebAssembly has one thread here, so the serial path is the only path.
    if !cfg!(feature = "parallel") || out.len() * in_dim < PARALLEL_MAC_THRESHOLD {
        for (i, o) in out.iter_mut().enumerate() {
            *o = dot(&w[i * in_dim..(i + 1) * in_dim], x);
        }
    } else {
        #[cfg(feature = "parallel")]
        {
            let rows_per_task = (MACS_PER_TASK / in_dim).max(1);
            out.par_chunks_mut(rows_per_task)
                .enumerate()
                .for_each(|(chunk, rows)| {
                    let first = chunk * rows_per_task;
                    for (offset, o) in rows.iter_mut().enumerate() {
                        let i = first + offset;
                        *o = dot(&w[i * in_dim..(i + 1) * in_dim], x);
                    }
                });
        }
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // Four partial sums so the compiler can keep several FMA chains in flight
    // without changing the result beyond f32 reduction-order noise.
    let mut s = [0.0f32; 4];
    let chunks = a.len() / 4;
    for c in 0..chunks {
        let i = c * 4;
        s[0] += a[i] * b[i];
        s[1] += a[i + 1] * b[i + 1];
        s[2] += a[i + 2] * b[i + 2];
        s[3] += a[i + 3] * b[i + 3];
    }
    let mut total = s[0] + s[1] + s[2] + s[3];
    for i in chunks * 4..a.len() {
        total += a[i] * b[i];
    }
    total
}

/// RMSNorm with an f32 accumulator, matching the reference implementation which
/// upcasts to f32 before normalising and multiplies by `weight.float()`.
pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

pub fn rms_norm_in_place(x: &mut [f32], weight: &[f32], eps: f32) {
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for i in 0..x.len() {
        x[i] = x[i] * scale * weight[i];
    }
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// In-place softmax over the whole slice, max-shifted for stability.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Rotary embedding, **interleaved** pairing: element `2i` is rotated against
/// `2i + 1`. This mirrors `view_as_complex(x.reshape(..., -1, 2))` in the
/// reference model, and is *not* the split-half (GPT-NeoX) convention.
pub fn apply_rope(vec: &mut [f32], pos: usize, theta: f32) {
    let head_dim = vec.len();
    for i in (0..head_dim).step_by(2) {
        let freq = 1.0 / theta.powf(i as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let (re, im) = (vec[i], vec[i + 1]);
        vec[i] = re * cos - im * sin;
        vec[i + 1] = re * sin + im * cos;
    }
}

/// Precomputed cos/sin table for positions `0..max_pos`, laid out as
/// `[pos][head_dim / 2 * 2]` pairs of `(cos, sin)`.
pub struct RopeTable {
    head_dim: usize,
    table: Vec<f32>,
}

impl RopeTable {
    pub fn new(head_dim: usize, max_pos: usize, theta: f32) -> Self {
        let half = head_dim / 2;
        let mut table = Vec::with_capacity(max_pos * half * 2);
        for pos in 0..max_pos {
            for i in 0..half {
                let freq = 1.0 / (theta as f64).powf((2 * i) as f64 / head_dim as f64);
                let angle = pos as f64 * freq;
                table.push(angle.cos() as f32);
                table.push(angle.sin() as f32);
            }
        }
        Self { head_dim, table }
    }

    pub fn apply(&self, vec: &mut [f32], pos: usize) {
        debug_assert_eq!(vec.len(), self.head_dim);
        let half = self.head_dim / 2;
        let row = &self.table[pos * half * 2..(pos + 1) * half * 2];
        for i in 0..half {
            let (cos, sin) = (row[i * 2], row[i * 2 + 1]);
            let (re, im) = (vec[i * 2], vec[i * 2 + 1]);
            vec[i * 2] = re * cos - im * sin;
            vec[i * 2 + 1] = re * sin + im * cos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matvec_when_identity_then_returns_input() {
        let w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let x = vec![2.0, 3.0, 4.0];
        let mut out = vec![0.0; 3];
        matvec(&mut out, &w, &x);
        assert_eq!(out, x, "identity matrix must reproduce the input vector");
    }

    #[test]
    fn test_softmax_when_uniform_then_equal_probabilities() {
        let mut x = vec![1.0, 1.0, 1.0, 1.0];
        softmax(&mut x);
        for v in &x {
            assert!((v - 0.25).abs() < 1e-6, "expected 0.25, got {v}");
        }
    }

    #[test]
    fn test_rope_table_when_pos_zero_then_identity() {
        let table = RopeTable::new(8, 4, 100_000.0);
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = v.clone();
        table.apply(&mut v, 0);
        assert_eq!(v, original, "position 0 rotation must be the identity");
    }

    #[test]
    fn test_rope_table_matches_direct_computation() {
        let table = RopeTable::new(64, 16, 100_000.0);
        for pos in [1usize, 7, 15] {
            let mut a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.3).collect();
            let mut b = a.clone();
            table.apply(&mut a, pos);
            apply_rope(&mut b, pos, 100_000.0);
            for (x, y) in a.iter().zip(&b) {
                assert!(
                    (x - y).abs() < 1e-5,
                    "table and direct RoPE disagree at pos {pos}: {x} vs {y}"
                );
            }
        }
    }

    #[test]
    fn test_rms_norm_when_unit_weight_then_unit_rms() {
        let x = vec![3.0, -4.0, 0.0, 5.0];
        let weight = vec![1.0; 4];
        let mut out = vec![0.0; 4];
        rms_norm(&mut out, &x, &weight, 1e-6);
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-4, "expected unit RMS, got {rms}");
    }
}
