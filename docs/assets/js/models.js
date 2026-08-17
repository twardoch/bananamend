// this_file: docs/assets/js/models.js
//
// The checkpoints that the demonstration page can load.
//
// Each entry names a repository on Hugging Face. The page downloads the four
// parts of the checkpoint from that repository, so any repository with the same
// four files works, including one of your own.
//
// `megabytes` is the size of the download. `memory` is what the engine holds
// while it runs, which for a quantized checkpoint is near the size of the file.

export const MODELS = [
  {
    repo: "fontlab/BananaMind-2-Nano-Chat-int8",
    label: "Nano · 8 bits",
    megabytes: 11,
    note: "The fastest download, and it answers like the original.",
    quality: "97.9% same next token",
    recommended: true,
  },
  {
    repo: "BananaMind/BananaMind-2-Nano-Chat",
    label: "Nano · 32-bit floats",
    megabytes: 40,
    note: "The original checkpoint. Every other Nano entry is measured against it.",
    quality: "the reference",
  },
  {
    repo: "fontlab/BananaMind-2-Nano-Chat-mixed",
    label: "Nano · mixed (3 ternary matrices)",
    megabytes: 11,
    note: "Ternary weights where a measurement showed the answers barely move.",
    quality: "96.8% same next token",
  },
  {
    repo: "fontlab/BananaMind-2-Nano-Chat-ternary",
    label: "Nano · ternary everywhere",
    megabytes: 6,
    note: "It does not work. It writes fragments and repeats them. Here to be seen.",
    quality: "22.1% same next token",
    research: true,
  },
  {
    repo: "fontlab/BananaMind-2-Mini-Chat-int8",
    label: "Mini · 8 bits",
    megabytes: 27,
    note: "A larger model, and still a small download.",
    quality: "96.8% same next token",
  },
  {
    repo: "BananaMind/BananaMind-2-Mini-Chat",
    label: "Mini · 32-bit floats",
    megabytes: 101,
    note: "The original Mini checkpoint.",
    quality: "the reference",
  },
  {
    repo: "fontlab/BananaMind-2-Pro-Preview-Chat-int8",
    label: "Pro · 8 bits",
    megabytes: 150,
    note: "The best answers of this family that a browser can hold.",
    quality: "100% same next token",
  },
  {
    repo: "fontlab/BananaMind-2-Pro-Preview-Chat-mixed",
    label: "Pro · mixed (12 ternary matrices)",
    megabytes: 147,
    note: "Ternary weights in twelve matrices of the 168.",
    quality: "90.5% same next token",
  },
  {
    repo: "fontlab/BananaMind-2-Pro-Preview-Chat-ternary",
    label: "Pro · ternary everywhere",
    megabytes: 69,
    note: "It does not work either, but it is less bad than Nano in ternary.",
    quality: "41.3% same next token",
    research: true,
  },
  {
    repo: "BananaMind/BananaMind-2-Pro-Preview-Chat",
    label: "Pro · 32-bit floats",
    megabytes: 556,
    note: "The engine needs approximately three times this size while it reads, so a browser may refuse it. The 8-bit copy is the safe choice.",
    quality: "the reference",
    heavy: true,
  },
];

// A checkpoint above this size makes a browser stop. The engine reads the file
// into memory and then builds the weights, so the peak is about three times the
// size of the download.
export const HEAVY_MEGABYTES = 500;

export function findModel(repo) {
  return MODELS.find((model) => model.repo === repo);
}
