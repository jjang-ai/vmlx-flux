# vmlx-flux Progress Tracker

**Last session**: 2026-04-13
**Branch**: `main`
**Last commit**: `fdc1f68` (lowercase-v rename + ARCHITECTURE.md)
**Build**: `swift build` ✅ clean · `swift test` ✅ **16/16** passing
**Repo**: https://github.com/jjang-ai/vmlx-flux

This file is the single source of truth for **what's done** and **what's
next** in vmlx-flux. Start here when resuming the project. The LLM/chat/
server side of vMLX is tracked separately in `../vllm-mlx/swift/PROGRESS.md`
— this doc covers **image + video generation only**.

For the "why does this code exist" architecture, see `ARCHITECTURE.md`.

---

## ✅ Done (committed + building + tested)

### Package scaffold
- [x] 4 SwiftPM products: `vMLXFlux` (umbrella), `vMLXFluxKit` (core),
      `vMLXFluxModels` (concrete), `vMLXFluxVideo` (Wan)
- [x] Sibling dep on `vmlx-swift-lm` for `JangLoader` / `JangConfig` /
      `TQDiskSerializer` reuse
- [x] Same `osaurus-ai/mlx-swift` fork as vmlx-swift-lm (single MLX runtime)
- [x] Lowercase-v naming (`vMLXFlux*`) to match the rest of the vMLX
      Swift rewrite — vMLX imports via `import vMLXFlux` directly
- [x] `README.md`, `ARCHITECTURE.md`, `PROGRESS.md` (this file)
- [x] `.gitignore` excludes `.build`, `.swiftpm`, `Package.resolved`

### Core runtime (`vMLXFluxKit`)
- [x] **`FluxModel.swift`** — capability protocols (`ImageGenerator`,
      `ImageEditor`, `ImageUpscaler`, `VideoGenerator`), `ModelKind`
      enum, `FluxError` types
- [x] **`Requests.swift`** — `ImageGenRequest` / `ImageEditRequest` /
      `UpscaleRequest` / `VideoGenRequest` structs + `ImageGenEvent` /
      `VideoGenEvent` streaming enums with `hfAuth` flag baked in
- [x] **`ModelRegistry.swift`** — name → loader dispatch, decentralized
      `static let _register` idiom, fuzzy lookup strips HF org prefix +
      `-Nbit` suffix + `.` normalization
- [x] **`FlowMatchScheduler.swift`** — rectified-flow Euler scheduler
      with resolution-dependent shift (`base=0.5`, `max=1.15`,
      `imageSeqLen` scales from 256→4096), `step(latent:velocity:stepIndex:)`
- [x] **`LatentSpace.swift`** — `initialNoise(width:height:layout:)` +
      two-layout support (`.fluxPatchified`, `.spatial`), top-level
      `patchify(_:patchSize:inChannels:)` + `unpatchify(...)` for
      Flux-style patch embed
- [x] **`MathOps.swift`** — `sinusoidalTimeEmbedding`, `RoPE2D`
      (cos/sin cache for H×W grid), `scaledDotProductAttention` with
      optional RoPE. Local `RMSNorm` removed; uses `MLXNN.RMSNorm`
      directly via `MLXFast.rmsNorm` kernel (~5× faster)
- [x] **`JangSupport.swift`** — `JangBridge.detect(at:)` / `.isJangModel` /
      `.loadConfig` — thin wrapper over vmlx-swift-lm's existing
      `JangLoader` so we don't re-port 533 lines of JANG v2 parsing
- [x] **`WeightLoader.swift`** — safetensors shard enumeration via
      `model.safetensors.index.json` (glob fallback), `MLX.loadArrays`
      per shard, merge into single `[String: MLXArray]` dict,
      `LoadedWeights(weights:jangConfig:)` return type
- [x] **`VAE.swift`** — full `AutoencoderKL` decoder for the Flux
      family (Flux1/Flux2/Qwen/FIBO/ZImage all share this):
      - `VAEGroupNorm` (32-group norm), `VAEResnetBlock`, `VAEAttnBlock`
        (single-head mid-block self-attention), `VAEUpsample`
      - `VAEDecoder` assembler with `128→256→512→512` channel
        progression reversed, conv_in → mid_block → 4 up_blocks → norm_out
      - `preprocessFluxLatent` (scale 0.3611 / shift 0.1159) + `postprocess`
      - `nchwToNhwc` / `nhwcToNchw` layout helpers for Conv2d bridging
- [x] **`FluxDiT.swift`** — full FLUX.1 transformer backbone:
      - `FluxModulation` returns `[ModTriple]` (named struct, not tuple)
      - `QKNorm` using hardware-accelerated `MLXNN.RMSNorm`
      - `FluxDoubleStreamBlock` — MM-DiT dual image+text attention,
        joint-concat softmax, two gates (attn, mlp)
      - `FluxSingleStreamBlock` — fused attention + parallel MLP in
        one residual
      - `splitQKV` helper (B,N,3·D) → 3× (B,H,N,D_head)
      - `FluxFinalLayer` — norm + modulated linear → patchified output
      - **Configs**: `.schnell` (19+38 blocks, no CFG), `.dev` (same
        topology + guidance embed), `.zImageTurbo` (dim=2048, 8+16
        blocks, 16 heads), `.flux2Klein` (dim=3072, 19+38, guidance)
      - `FluxDiTModel` full assembler: img_in → time/vector/guidance
        projections → txt_in → double blocks → concat → single
        blocks → drop text → final_layer
      - **40-line weight key-map docblock** listing every Black Forest
        Labs checkpoint key with its Swift property counterpart
- [x] **`ImageIO.swift`** — `@MainActor writePNG` via `NSBitmapImageRep`
      with clamp + scale + dtype conversion from MLX → RGB bytes

### Concrete models (`vMLXFluxModels`)
All 11 model files registered with correct canonical names, default
steps, default guidance, and loader closures. Only ZImage has real
pipeline wiring; everything else throws `notImplemented` pending per-model
port work.

- [x] **`Flux1/Flux1.swift`** — Flux1Schnell, Flux1Dev, Flux1Kontext, Flux1Fill
- [x] **`Flux2Klein/Flux2Klein.swift`** — Flux2Klein, Flux2KleinEdit
- [x] **`ZImage/ZImage.swift`** — ✅ **end-to-end wired**:
      - Loads weights via `WeightLoader` at init
      - Uses `FluxDiTConfig.zImageTurbo` preset (not the oversized Schnell)
      - Allocates noise latent at `transformer.config.inChannels` (16,
        not 4 — this was a runtime bug fixed in audit pass 2)
      - Real sampling loop: patchify → `FluxDiTModel(imgPatched:txt:
        pooledClip:timestep:guidance:rope:)` → unpatchify → Euler step
      - Real VAE decode via `VAEDecoder.preprocessFluxLatent` → `vae(...)` →
        `postprocess`
      - Real PNG write via `ImageIO.writePNG`
      - Progress events with ETA, cancellation check per step,
        HF auth error detection (401/403 → `hfAuth=true`)
      - Text encoders are zero-tensor stubs (T5/CLIP not ported yet)
      - Module weights are random-init (no real safetensors load yet)
- [x] **`QwenImage/QwenImage.swift`** — QwenImage + QwenImageEdit stubs
- [x] **`FIBO/FIBO.swift`** — scaffold
- [x] **`SeedVR2/SeedVR2.swift`** — upscale scaffold
- [x] **`Exports.swift`** — `VMLXFluxModels.registerAll()` force-registers
      every model via `_register` statics

### Video runtime (`vMLXFluxVideo`)
- [x] **`WanVAE3D.swift`** — 3D causal VAE decoder:
      - `CausalConv3d` shim (collapses (B,C,T,H,W) → (B·T,C,H,W), runs
        Conv2d, un-collapses) until mlx-swift ships real Conv3d
      - `VAEGroupNorm3D`, `WanResBlock3D`, `WanUpsample3D` with optional
        temporal 2× upsample
      - `WanVAEDecoder` at `96→192→384→384` reversed
      - Wan scale/shift constants (0.2 / 0.0)
- [x] **`WanDiT.swift`** — Wan transformer block + model:
      - `WanDiTBlock` (self-attention + T5 cross-attention + gated MLP
        with AdaLN modulation)
      - `WanDiTModel` assembler
      - **Configs**: `.wan21_1_3B` (dim=1024, 24 layers, 8 heads),
        `.wan21_14B` (dim=1536, 30 layers, 12 heads), `.wan22` (same
        topology as 14B pending spec)
- [x] **`WanVideoIO.swift`** — `@MainActor writePNG` frames + full
      H.264 MP4 writer via `AVAssetWriter` + `AVAssetWriterInputPixelBufferAdaptor`
      with ARGB pixel format, per-frame `CVPixelBuffer` alloc + row-byte copy
- [x] **`WANModel.swift`** — **full end-to-end pipeline**:
      - Scheduler built at video-length `imageSeqLen = max(256, patched_spatial × patched_temporal)`
      - 5D noise latent (1, 16, T/4, H/8, W/8) with optional seed
      - 8-axis `patchify` / `unpatchify` helpers inline
      - Scheduler loop: patchify → `WanDiTModel(video, txt, t)` →
        unpatchify → Euler → progress event
      - `WanVAEDecoder.preprocessLatent` → `vae(...)` → `postprocess`
      - `WanVideoIO.writeMP4` with requested fps
      - `.completed(url, seed, fps, frameCount)` event
      - Both `wan-2.1` and `wan-2.2` registered in `VMLXFluxVideo.registerAll()`

### Bridge (lives in vMLX, not here — documented)
- [x] `vMLX/swift/Package.swift` declares `vmlx-flux` as a local path
      dep + `vMLXFlux` as a product dependency of `vMLXEngine`
- [x] `vMLX/swift/Sources/vMLXEngine/FluxBackend.swift` bridge:
      - Lazy-creates `vMLXFlux.FluxEngine` on first image call
      - Converts `vMLXEngine.ImageGenSettings` → `vMLXFlux.ImageGenRequest`
      - Fans events via `FluxJobBridge` (multi-subscriber)
      - `bridgeEvent` translates between the two `ImageGenEvent` enums
        (vmlx-flux has separate `.step` + `.preview`; vMLX has unified
        `.step(step:total:preview:)`)

### Tests (16/16 passing)
- [x] **`RegistryTests.swift`** — 8 tests:
      - All 6 image gen models registered
      - All 4 image edit models registered
      - Upscale model registered
      - Video stubs registered
      - Fuzzy lookup strips HF prefix + `-Nbit` suffix
      - Canonical lookup
      - `Engine.load` fails on missing weights
      - `Engine.generate` requires load
- [x] **`ShapeTests.swift`** — 8 pure-Swift tests (no MLX ops to avoid
      the "Failed to load the default metallib" crash in the test binary):
      - Flow scheduler sigmas monotonic decrease + bounds
      - Timesteps = sigmas × 1000 linkage
      - Resolution-dependent shift + endpoint clamping
      - `FluxDiTConfig` preset sanity (Schnell no CFG, Dev has CFG,
        zImageTurbo strictly smaller than Schnell)
      - `WanDiTConfig` preset sanity (1.3B < 14B dim, patch defaults)
      - VAE Flux scale/shift constants locked at 0.3611 / 0.1159
      - Wan VAE scale/shift locked at 0.2 / 0.0

---

## ❌ TODO — prioritized

### 🔴 P0 — blocking real output

1. **`.safetensors` key-mapped weight loader**
   - The `WeightLoader` currently returns `[String: MLXArray]` but
     nothing applies it to the module tree via `Module.update(parameters:)`
   - Needs a per-model key-map translator from BFL checkpoint keys
     (`double_blocks.{i}.img_mod.lin.weight`) to Swift property paths
     (`doubleBlocks[i].imgMod.linear.weight`)
   - Start with Flux1 Schnell as the reference implementation — the
     40-line docblock on `FluxDiTModel` has the full mapping table
   - Alternative: add `@ModuleInfo(key: "...")` decorators to every
     module property so mlx-swift's reflection-based loader works
     directly. Cleaner but requires rewriting every block class

2. **T5-XXL text encoder port**
   - Used by Flux1, Flux2, Qwen, FIBO, Wan — shared across the whole
     image + video stack, biggest single payoff
   - `swift-transformers` has tokenizer support but the T5 encoder
     forward pass needs porting from Python `transformers` or reusing
     mlx-examples' T5
   - Goes in `vMLXFluxKit/TextEncoders/T5XXL.swift`
   - Currently `txtEmb = MLXArray.zeros([1, 256, 4096])` in ZImage +
     WANModel, so every prompt produces the same (random-weight)
     output regardless of text input

3. **CLIP-L text encoder port** (Flux1 only)
   - Provides the pooled conditioning vector that feeds `vectorIn0`
   - Simpler than T5 — 12 layers vs 24, ~240M params
   - Can reuse swift-transformers' CLIP if available

### 🟡 P1 — correctness gaps that matter after P0 lands

4. **3-axis RoPE for Flux**
   - `FluxDoubleStreamBlock` + `FluxSingleStreamBlock` both have
     `_ = rope // TODO: split q/k, rope image half, reassemble`
   - Real Flux uses a 3-axis RoPE over (time=0 stub, H, W) that
     concatenates position IDs for the image stream while text tokens
     get zero positional rotation
   - Without this, positional information is completely lost — every
     image token is interchangeable

5. **FluxDiT Sendable conformance cleanup**
   - Current warnings about `MLXArray` non-Sendable in `RoPE2D` and
     `ModTriple` are suppressed by dropping `Sendable`. Long-term the
     upstream `MLXArray` should gain `@unchecked Sendable` (or we
     wrap it). Blocks Swift 6 language mode adoption.

6. **MXTQ PRNG bit-validation**
   - `vmlx-swift-lm/.../NumPyPCG64.swift` is a best-effort Swift port
     of NumPy's `default_rng(seed).choice([-1,1], dim)`
   - Needs a test that compares against a reference NumPy output
     before trusting it for production MXTQ weight decode
   - Without this, any MXTQ-quantized JANG model will load with
     garbage signs and produce noise

7. **`VMLXFluxModels.registerAll()` / `VMLXFluxVideo.registerAll()`
   symbol namespace collision**
   - These are top-level enum names that Swift auto-tuple-labeling
     has been known to confuse with the module name. Verify nothing
     in the bridge relies on `vMLXFluxModels` as both a module and
     a symbol.

### 🟢 P2 — per-model ports (after T5 + CLIP land)

Each of these needs: config parse from `config.json`, weight key-map
for the checkpoint layout, real generation loop body (the scheduler +
VAE machinery is shared via `vMLXFluxKit`).

8. **Flux1 Schnell** — simplest variant, 4 steps, no CFG, single best
   proof-of-concept to nail first
9. **Flux1 Dev** — adds guidance embedding to Schnell's pipeline
10. **Flux1 Kontext** — image edit via prompt only, no mask
11. **Flux1 Fill** — inpaint / outpaint via mask
12. **Flux2 Klein** — single-encoder variant
13. **Flux2 Klein Edit**
14. **Qwen-Image** — Alibaba's architecture, different tokenizer
15. **Qwen-Image-Edit**
16. **FIBO**
17. **SeedVR2** — super-resolution / upscale (different arch entirely)

### 🔵 P3 — video (Wan)

18. **Wan 2.1 / Wan 2.2 real weight loading**
    - `WANModel.swift` has the full pipeline scaffolded with random
      weights. Real weights need a key-map translator
19. **Real `Conv3d`** — currently `CausalConv3d` is a shim that
    collapses time into batch and runs `Conv2d`. Upstream mlx-swift
    hasn't shipped Conv3d yet. When it does, replace the shim.
20. **Windowed / separable attention for Wan**
    - Current `WanDiTBlock` does O(N²) self-attention over the full
      (T/4 × H/8 × W/8) video sequence — 100k+ tokens for 5s @ 720p
    - Real Wan uses windowed attention or temporal-spatial separation
    - Doesn't scale past 3-4 sec clips on an M-series machine

### 🟤 P4 — test infrastructure + DX

21. **Runtime shape validation that touches MLX**
    - Current `ShapeTests.swift` is pure-Swift only because the test
      binary lacks `default.metallib`
    - Options: (a) set MLX to CPU-only stream via
      `Device.setDefault(device: Device(.cpu))` + pray the
      metallib load happens lazily, (b) copy `default.metallib` into
      the test bundle as a resource, (c) build a runtime smoke-test
      CLI tool that loads a model + runs one forward pass + prints
      shape info for manual verification
    - The CLI tool option is probably easiest

22. **`@ModuleInfo(key: ...)` decorators on every module property**
    - Would half-solve the weight-loading problem by giving
      `Module.update(parameters:)` the right key names
    - Ugly because every field needs one — prefer an explicit remap
      table if we go that route

23. **ImageGenSettings.numImages > 1 support**
    - Currently `LatentSpace.initialNoise(batchSize: 1)` hard-codes
      batch = 1. Real numImages support means batching through the
      transformer + VAE + writing N PNGs

---

## 📋 Audit checks (passed at session close)

| Check | Status | Details |
|---|---|---|
| `swift build` clean | ✅ | 0 errors, warnings only from MLXArray non-Sendable (known) |
| `swift test` all pass | ✅ | 16/16 (8 RegistryTests + 8 ShapeTests) |
| vmlx-flux ↔ vmlx-swift-lm dep resolves | ✅ | sibling path `../vmlx-swift-lm` works |
| vmlx-flux ↔ mlx-swift dep resolves | ✅ | `osaurus-ai/mlx-swift @ osaurus-0.31.3` |
| Module naming convention | ✅ | all lowercase-v: `vMLXFlux*` matches vMLX side |
| No stale `VMLXFlux*` references in source | ✅ | `grep -rln "import VMLXFlux"` returns empty |
| `ARCHITECTURE.md` current | ✅ | 300+ lines, covers bridge location + flow |
| `PROGRESS.md` current | ✅ | this file |
| `README.md` | ✅ | usage example, architecture diagram, status notes |
| `.gitignore` excludes build artifacts | ✅ | `.build/`, `.swiftpm/`, `Package.resolved` |
| All commits pushed to origin/main | ✅ | last = `fdc1f68` |
| Weight key-map docblock on FluxDiTModel | ✅ | 40 lines, every BFL key mapped |
| Known deferred items documented | ✅ | marked TODO in code + listed in ARCHITECTURE.md §"Known deferred items" |

---

## 🗂 Session history (commits)

| Commit | Summary |
|---|---|
| `399af53` | Initial scaffold: 4 targets, stubs throw notImplemented, no real math |
| `d9b1feb` | Real math: FlowMatchEulerScheduler, LatentSpace, JangSupport bridge, WeightLoader, MathOps (RoPE, attention), ImageIO.writePNG, ZImage end-to-end with placeholder velocity/VAE |
| `2a69871` | VAE decoder (Flux family) + Flux DiT transformer blocks (MMDiT double + fused single) + full Wan 2.x video pipeline (WanVAE3D + WanDiT + WanVideoIO MP4 writer + WANModel) |
| `1911cc4` | Audit pass 1: FluxModulation tuple bug → ModTriple struct, ZImage never called its own transformer (fixed), layout helpers renamed, Z-Image Turbo config preset added, 8 new shape tests |
| `a0a1258` | Audit pass 2: RMSNorm → MLXNN hardware path, ZImage latent channel count fixed (was 4, needed 16) |
| `9981196` | Audit pass 3: weight key-map docblock + WanVideoIO comment drift |
| `fdc1f68` | Lowercase-v module rename (VMLXFlux* → vMLXFlux*) + ARCHITECTURE.md |

---

## 🧭 Resuming next session

1. Read this file top to bottom
2. Read `ARCHITECTURE.md` §"Integration with vMLX" to refresh on where
   the bridge lives (spoiler: in vMLX, not here)
3. Pick a P0 item — probably #1 (real weight loader) or #2 (T5-XXL port)
4. Verify baseline: `cd vmlx-flux && swift build && swift test` should
   still be `Build complete!` + `16/16 passing`
5. Update this file + `ARCHITECTURE.md` as work lands
6. `git push` every logical checkpoint — the commit history is the
   primary handoff doc

The LLM/engine/API/UI side of the vMLX Swift rewrite is tracked
separately at `../vllm-mlx/swift/PROGRESS.md` and is owned by a
different agent branch.
