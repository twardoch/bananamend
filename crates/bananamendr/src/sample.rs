//! this_file: crates/bananamendr/src/sample.rs
//!
//! Logit post-processing and sampling. `temperature = 0` means greedy (argmax),
//! which is the mode used for parity testing against `transformers`.

/// xorshift64* — small, deterministic, and enough for token sampling.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// `0.0` selects greedy decoding.
    pub temperature: f32,
    /// `0` disables top-k.
    pub top_k: usize,
    /// `1.0` disables nucleus filtering.
    pub top_p: f32,
    /// `1.0` disables the penalty; values above 1 discourage repeats.
    pub repetition_penalty: f32,
    /// How many of the most recent tokens the repetition penalty considers.
    pub repetition_window: usize,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            repetition_window: 64,
            seed: 0,
        }
    }
}

impl SamplingConfig {
    pub fn greedy() -> Self {
        Self::default()
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Applies the repetition penalty in place, following the common convention:
/// positive logits are divided by the penalty, negative logits multiplied.
pub fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32, window: usize) {
    if penalty == 1.0 || history.is_empty() || window == 0 {
        return;
    }
    let start = history.len().saturating_sub(window);
    for &token in &history[start..] {
        let i = token as usize;
        if i >= logits.len() {
            continue;
        }
        logits[i] = if logits[i] > 0.0 {
            logits[i] / penalty
        } else {
            logits[i] * penalty
        };
    }
}

/// Samples a token id. `logits` is modified in place (temperature scaling,
/// penalties) — pass a scratch copy if the caller needs the raw values.
pub fn sample(logits: &mut [f32], history: &[u32], config: &SamplingConfig, rng: &mut Rng) -> u32 {
    apply_repetition_penalty(
        logits,
        history,
        config.repetition_penalty,
        config.repetition_window,
    );
    if config.is_greedy() {
        return argmax(logits);
    }

    let mut candidates: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v / config.temperature))
        .collect();
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    if config.top_k > 0 && config.top_k < candidates.len() {
        candidates.truncate(config.top_k);
    }

    let max = candidates[0].1;
    let mut sum = 0.0;
    for c in candidates.iter_mut() {
        c.1 = (c.1 - max).exp();
        sum += c.1;
    }
    for c in candidates.iter_mut() {
        c.1 /= sum;
    }

    if config.top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut keep = candidates.len();
        for (i, c) in candidates.iter().enumerate() {
            cumulative += c.1;
            if cumulative >= config.top_p {
                keep = i + 1;
                break;
            }
        }
        candidates.truncate(keep);
        let total: f32 = candidates.iter().map(|c| c.1).sum();
        for c in candidates.iter_mut() {
            c.1 /= total;
        }
    }

    let target = rng.next_f32();
    let mut cumulative = 0.0;
    for &(id, p) in &candidates {
        cumulative += p;
        if target < cumulative {
            return id;
        }
    }
    candidates.last().expect("candidate list is never empty").0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax_when_ties_then_returns_first() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn test_sample_when_temperature_zero_then_greedy() {
        let mut logits = vec![0.1, 0.9, 0.4];
        let mut rng = Rng::new(1);
        let id = sample(&mut logits, &[], &SamplingConfig::greedy(), &mut rng);
        assert_eq!(id, 1, "temperature 0 must pick the argmax");
    }

    #[test]
    fn test_sample_when_top_k_one_then_deterministic() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            ..SamplingConfig::default()
        };
        let mut rng = Rng::new(7);
        for _ in 0..8 {
            let mut logits = vec![0.1, 0.9, 0.4];
            assert_eq!(sample(&mut logits, &[], &config, &mut rng), 1);
        }
    }

    #[test]
    fn test_repetition_penalty_when_token_seen_then_logit_reduced() {
        let mut logits = vec![2.0, -2.0];
        apply_repetition_penalty(&mut logits, &[0, 1], 2.0, 64);
        assert_eq!(logits, vec![1.0, -4.0]);
    }

    #[test]
    fn test_rng_when_seeded_then_reproducible() {
        let a: Vec<f32> = (0..5).map(|_| Rng::new(42).next_f32()).collect();
        let b: Vec<f32> = (0..5).map(|_| Rng::new(42).next_f32()).collect();
        assert_eq!(a, b);
        let mut rng = Rng::new(42);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "sample out of range: {v}");
        }
    }
}
