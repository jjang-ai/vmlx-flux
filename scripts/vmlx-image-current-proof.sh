#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

STAMP="${VMLX_IMAGE_PROOF_STAMP:-$(date -u +%Y-%m-%dT%H%M%SZ)}"
DOC_PREFIX=""
if [[ -f "$ROOT/docs/OSAURUS_IMAGE_UI_MANIFEST.json" ]]; then
  DOC_PREFIX="docs/"
fi

ART_ROOT="${VMLX_IMAGE_PROOF_ARTIFACT_ROOT:-${DOC_PREFIX}local/vmlx-flux-probes/${STAMP}}"
OUT_ROOT="${VMLX_IMAGE_PROOF_OUTPUT_ROOT:-${DOC_PREFIX}local/vmlx-flux-outputs/${STAMP}}"
MODEL_ROOT="${VMLX_IMAGE_MODEL_ROOT:-}"
WIDTH="${VMLX_IMAGE_PROOF_WIDTH:-512}"
HEIGHT="${VMLX_IMAGE_PROOF_HEIGHT:-512}"
SEED="${VMLX_IMAGE_PROOF_SEED:-7}"
IDEOGRAM_SEED="${VMLX_IMAGE_PROOF_IDEOGRAM_SEED:-103437}"
Z_STEPS="${VMLX_IMAGE_PROOF_Z_STEPS:-8}"
FLUX_STEPS="${VMLX_IMAGE_PROOF_FLUX_STEPS:-4}"
QWEN_IMAGE_STEPS="${VMLX_IMAGE_PROOF_QWEN_IMAGE_STEPS:-20}"
QWEN_EDIT_STEPS="${VMLX_IMAGE_PROOF_QWEN_EDIT_STEPS:-20}"
IDEOGRAM_STEPS="${VMLX_IMAGE_PROOF_IDEOGRAM_STEPS:-20}"
SKIP_BUILD="${VMLX_IMAGE_PROOF_SKIP_BUILD:-0}"
RUN_CONTRACT_CHECK="${VMLX_IMAGE_PROOF_CONTRACT_CHECK:-1}"
SUMMARY_ONLY="${VMLX_IMAGE_PROOF_SUMMARY_ONLY:-0}"

mkdir -p "$ART_ROOT" "$OUT_ROOT"

if [[ "$SKIP_BUILD" != "1" && "$SUMMARY_ONLY" != "1" ]]; then
  swift build --product vmlxflux-probe
fi

resolve_probe() {
  if [[ -n "${VMLX_FLUX_PROBE:-}" ]]; then
    printf '%s\n' "$VMLX_FLUX_PROBE"
    return
  fi

  local candidate
  for candidate in \
    ".build/arm64-apple-macosx/debug/vmlxflux-probe" \
    ".build/debug/vmlxflux-probe"
  do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  printf 'vmlx-image-current-proof: vmlxflux-probe executable not found after build\n' >&2
  return 1
}

if [[ "$SUMMARY_ONLY" != "1" ]]; then
  PROBE="$(resolve_probe)"
fi
run_probe() {
  local label="$1"
  shift
  local artifacts="${ART_ROOT}/${label}"
  local outputs="${OUT_ROOT}/${label}"
  mkdir -p "$artifacts" "$outputs"
  printf '\n== %s ==\n' "$label"
  if [[ -n "$MODEL_ROOT" ]]; then
    "$PROBE" --root "$MODEL_ROOT" "$@" --artifacts "$artifacts" --output-dir "$outputs"
  else
    "$PROBE" "$@" --artifacts "$artifacts" --output-dir "$outputs"
  fi
}

APPLE_PROMPT="a red apple on a plain white background, centered, clean product photo"
MOUNTAIN_PROMPT="a blue mountain landscape under a golden sun, watercolor"
EDIT_APPLE_PROMPT="turn the apple blue while keeping it centered on a plain white background"
EDIT_PEAR_PROMPT="turn the apple into a green pear on a plain white background"
IDEOGRAM_APPLE_PROMPT="a clean vector icon of one red apple centered on a pure white background, no text, no letters, no watermark"
IDEOGRAM_MOUNTAIN_PROMPT="a clean vector icon of blue mountains and a yellow sun centered on a pure white background, no text, no letters, no watermark"

if [[ "$SUMMARY_ONLY" != "1" ]]; then
  run_probe "status-load-matrix" \
    --matrix --no-generate

  run_probe "zimage-4bit-gen" \
    --model Z-Image-Turbo-mflux-4bit --generate \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$Z_STEPS" \
    --turn "$APPLE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$APPLE_PROMPT"

  run_probe "zimage-8bit-gen" \
    --model Z-Image-Turbo-mflux-8bit --generate \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$Z_STEPS" \
    --turn "$APPLE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$APPLE_PROMPT"

  run_probe "flux-schnell-4bit-gen" \
    --model FLUX.1-schnell-mflux-4bit --generate \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$FLUX_STEPS" \
    --turn "$APPLE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$APPLE_PROMPT"

  run_probe "flux-schnell-8bit-gen" \
    --model FLUX.1-schnell-mflux-8bit --generate \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$FLUX_STEPS" \
    --turn "$APPLE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$APPLE_PROMPT"

  run_probe "qwen-image-4bit-gen" \
    --model qwen-image-mflux-4bit --generate \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$QWEN_IMAGE_STEPS" \
    --turn "$APPLE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$APPLE_PROMPT"

  run_probe "qwen-image-8bit-gen" \
    --model qwen-image-mflux-8bit --generate \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$QWEN_IMAGE_STEPS" \
    --turn "$APPLE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$APPLE_PROMPT"

  QWEN_SOURCE_IMAGE="${VMLX_IMAGE_PROOF_SOURCE_IMAGE:-}"
  if [[ -z "$QWEN_SOURCE_IMAGE" ]]; then
    QWEN_SOURCE_IMAGE="$(
      node - "$ART_ROOT/qwen-image-8bit-gen/qwen-image-mflux-8bit-load.json" <<'NODE'
const fs = require("fs");
const payload = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const turn = (payload.generation_turns || []).find((entry) => entry.turn === 1 && entry.status === "completed");
const output = turn?.image_diagnostics?.path || turn?.output;
if (!output) process.exit(2);
process.stdout.write(output);
NODE
    )"
  fi

  if [[ ! -f "$QWEN_SOURCE_IMAGE" ]]; then
    printf 'vmlx-image-current-proof: qwen edit source image missing: %s\n' "$QWEN_SOURCE_IMAGE" >&2
    exit 1
  fi

  run_probe "qwen-edit-q4-gen" \
    --model Qwen-Image-Edit-mflux-q4 --edit --source-image "$QWEN_SOURCE_IMAGE" \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$QWEN_EDIT_STEPS" \
    --turn "$EDIT_APPLE_PROMPT" --turn "$EDIT_APPLE_PROMPT" --turn "$EDIT_PEAR_PROMPT"

  run_probe "qwen-edit-q8-gen" \
    --model Qwen-Image-Edit-mflux-q8 --edit --source-image "$QWEN_SOURCE_IMAGE" \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$QWEN_EDIT_STEPS" \
    --turn "$EDIT_APPLE_PROMPT" --turn "$EDIT_APPLE_PROMPT" --turn "$EDIT_PEAR_PROMPT"

  run_probe "ideogram-fp8-gen" \
    --model ideogram-4-fp8 --generate \
    --seed "$IDEOGRAM_SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$IDEOGRAM_STEPS" \
    --turn "$IDEOGRAM_APPLE_PROMPT" --turn "$IDEOGRAM_MOUNTAIN_PROMPT" --turn "$IDEOGRAM_APPLE_PROMPT"

  run_probe "ideogram-nf4-gen" \
    --model ideogram-4-nf4 --generate \
    --seed "$IDEOGRAM_SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$IDEOGRAM_STEPS" \
    --turn "$IDEOGRAM_APPLE_PROMPT" --turn "$IDEOGRAM_MOUNTAIN_PROMPT" --turn "$IDEOGRAM_APPLE_PROMPT"
fi

SUMMARY="$ART_ROOT/current-proof-summary.json"
node - "$ART_ROOT" "$OUT_ROOT" "$SUMMARY" <<'NODE'
const fs = require("fs");
const path = require("path");

const artifactRoot = process.argv[2];
const outputRoot = process.argv[3];
const summaryPath = process.argv[4];

const runs = [
  ["zimage-4bit-gen", "Z-Image-Turbo-mflux-4bit-load.json", "generation_turns"],
  ["zimage-8bit-gen", "Z-Image-Turbo-mflux-8bit-load.json", "generation_turns"],
  ["flux-schnell-4bit-gen", "FLUX.1-schnell-mflux-4bit-load.json", "generation_turns"],
  ["flux-schnell-8bit-gen", "FLUX.1-schnell-mflux-8bit-load.json", "generation_turns"],
  ["qwen-image-4bit-gen", "qwen-image-mflux-4bit-load.json", "generation_turns"],
  ["qwen-image-8bit-gen", "qwen-image-mflux-8bit-load.json", "generation_turns"],
  ["qwen-edit-q4-gen", "Qwen-Image-Edit-mflux-q4-load.json", "edit_turns"],
  ["qwen-edit-q8-gen", "Qwen-Image-Edit-mflux-q8-load.json", "edit_turns"],
  ["ideogram-fp8-gen", "ideogram-4-fp8-load.json", "generation_turns"],
  ["ideogram-nf4-gen", "ideogram-4-nf4-load.json", "generation_turns"],
];

let failed = false;
const rows = runs.map(([label, file, key]) => {
  const filePath = path.join(artifactRoot, label, file);
  const payload = JSON.parse(fs.readFileSync(filePath, "utf8"));
  const turns = payload[key] || [];
  const statuses = turns.map((turn) => turn.status);
  const shas = turns.map((turn) => turn.image_diagnostics?.sha256 || null);
  const outputs = turns.map((turn) => turn.image_diagnostics?.path || turn.output || null);
  const completed = statuses.length === 3 && statuses.every((status) => status === "completed");
  const repeatIndex = key === "edit_turns" ? 1 : 2;
  const sensitiveIndex = key === "edit_turns" ? 2 : 1;
  const deterministicRepeat = Boolean(shas[0] && shas[repeatIndex] && shas[0] === shas[repeatIndex]);
  const promptSensitive = Boolean(shas[0] && shas[sensitiveIndex] && shas[0] !== shas[sensitiveIndex]);
  const status = completed && deterministicRepeat && promptSensitive ? "passed" : "failed";
  if (status !== "passed") failed = true;
  return {
    label,
    artifact: filePath,
    load_status: payload.load_status,
    turn_key: key,
    statuses,
    shas,
    outputs,
    repeat_turns: [1, repeatIndex + 1],
    prompt_sensitive_turns: [1, sensitiveIndex + 1],
    deterministic_repeat: deterministicRepeat,
    prompt_sensitive: promptSensitive,
    status,
  };
});

const matrixPath = path.join(artifactRoot, "status-load-matrix", "compatibility-matrix.json");
const matrix = JSON.parse(fs.readFileSync(matrixPath, "utf8"));
const unloaded = (matrix.rows || []).filter((row) => row.load_status !== "loaded");
if (unloaded.length > 0) failed = true;

const summary = {
  status: failed ? "failed" : "passed",
  generated_at: new Date().toISOString(),
  artifact_root: artifactRoot,
  output_root: outputRoot,
  matrix: {
    artifact: matrixPath,
    model_count: matrix.model_count,
    loaded_count: (matrix.rows || []).filter((row) => row.load_status === "loaded").length,
    unloaded: unloaded.map((row) => row.directory_name),
  },
  rows,
  visual_gate: "View generated PNGs before claiming visual quality or Osaurus release readiness.",
  osaurus_gate: "Osaurus HTTP/UI bridge proof is outside this CLI runner and remains required before app-side readiness claims.",
};

fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2) + "\n");
console.log(`current proof summary: ${summaryPath}`);
console.log(`status=${summary.status} matrix_loaded=${summary.matrix.loaded_count}/${summary.matrix.model_count}`);
for (const row of rows) {
  console.log(`${row.label}: ${row.status} repeat=${row.deterministic_repeat} sensitive=${row.prompt_sensitive} sha=${row.shas.join(",")}`);
}
if (failed) process.exit(1);
NODE

if [[ "$RUN_CONTRACT_CHECK" == "1" && -x "$ROOT/scripts/vmlx-image-openapi-manifest-check.sh" ]]; then
  "$ROOT/scripts/vmlx-image-openapi-manifest-check.sh"
fi

printf '\nProof artifacts: %s\n' "$ART_ROOT"
printf 'Proof outputs:   %s\n' "$OUT_ROOT"
printf 'Summary:         %s\n' "$SUMMARY"
printf 'PARTIAL until generated PNGs are visually inspected and Osaurus HTTP/UI bridge proof exists.\n'
