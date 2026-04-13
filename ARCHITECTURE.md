# vmlx-flux Architecture

**Last updated**: 2026-04-13 (post-rename + integration-move)

## Where the code lives

vmlx-flux is a **standalone Swift package** at
`https://github.com/jjang-ai/vmlx-flux` that provides image + video
generation for the vMLX Swift rewrite. It's imported by vMLX as a local
sibling dependency — `../../../vmlx-flux` relative to vMLX's
`swift/Package.swift`.

vmlx-flux itself depends on two things:

1. `osaurus-ai/mlx-swift` on branch `osaurus-0.31.3` — same fork as
   vmlx-swift-lm, so MLX is shared across all three packages
2. `../vmlx-swift-lm` — reused for `JangConfig` / `JangLoader` / the
   `TQDiskSerializer` so we don't re-port 533 lines of JANG v2 parsing

## Module naming convention

**All products use the lowercase-`v` convention** to match the rest of
the vMLX Swift rewrite (`vMLXEngine`, `vMLXServer`, `vMLXTheme`,
`vMLXApp`, `vMLXCLI`):

| Library | Import | Purpose |
|---|---|---|
| `vMLXFlux` | umbrella | One-stop `import vMLXFlux` — re-exports the 3 sub-modules |
| `vMLXFluxKit` | core | Schedulers, VAE, RoPE, latent patchify, weight loader, JANG bridge, image IO, FluxDiT transformer block |
| `vMLXFluxModels` | concrete models | Flux1 Schnell/Dev/Kontext/Fill, Flux2 Klein, ZImage, QwenImage, FIBO, SeedVR2 |
| `vMLXFluxVideo` | video (WAN) | Wan 2.1/2.2 transformer + 3D causal VAE + MP4 writer |

Historical note: an earlier commit used uppercase-V (`VMLXFlux`). It was
renamed on 2026-04-13 to match vMLX's convention so `import vMLXFlux`
reads consistently with `import vMLXEngine`. The git history shows the
rename via case-insensitive tmp-folder shuffles.

## Integration with vMLX (where the bridge lives)

**The vMLX ↔ vmlx-flux bridge lives in the vMLX repo, not here.**
Specifically:

- `vMLX/swift/Package.swift` declares vmlx-flux as a local path dep
  and adds `vMLXFlux` as a product dependency of the `vMLXEngine` target
- `vMLX/swift/Sources/vMLXEngine/FluxBackend.swift` is the bridge file
  that:
  1. Holds a lazy-loaded `vMLXFlux.FluxEngine` instance on the vMLX
     `Engine` actor (stored as `Any?` to avoid forcing every file that
     imports `vMLXEngine` to also import `vMLXFlux`)
  2. Implements `Engine.generateImage(prompt:model:settings:)` which
     routes to the flux backend, loads the requested model if it's not
     already resident, drains the `AsyncThrowingStream<ImageGenEvent>`,
     and returns the final image URL
  3. Implements `Engine.editImage(prompt:source:mask:strength:settings:)`
     (currently routes to `vmlx-flux` editor paths — each editor model
     still throws `notImplemented` until its own port lands)
  4. Implements `Engine.imageGenStream(jobId:)` which fans out the
     bridged events from a per-job `FluxJobBridge` so the UI can
     subscribe without holding a direct flux reference
  5. Translates between `vMLXFlux.ImageGenEvent` (which has separate
     `.step` and `.preview` cases) and `vMLXEngine.ImageGenEvent`
     (which has a single `.step(step:total:preview:)` case) — shapes
     differ because the two packages have different evolution histories

**vmlx-flux itself has zero knowledge of vMLX.** It only knows its own
types (`FluxEngine`, `ImageGenRequest`, `ImageGenEvent`, etc.). This
means vmlx-flux can theoretically be used by any Swift app without
pulling in vMLX's Hummingbird server, chat UI, download manager, etc.

## Why the bridge lives in vMLX and not vmlx-flux

The obvious question: why doesn't vmlx-flux depend on vMLX and expose a
ready-to-plug-in `Engine.imageBackend` extension directly?

Because:
- **Dependency direction** — vMLX depends on vmlx-flux, not the other
  way around. Reversing this would create a circular dep that SwiftPM
  won't resolve.
- **vmlx-flux is reusable** — another app (or a future CLI tool) can
  use `vMLXFlux.FluxEngine` without any vMLX baggage. The bridge lives
  on the vMLX side so that coupling is one-directional.
- **UI types belong to the app, not the engine** — `ImageGenSettings`
  on the vMLX side has app-specific fields (e.g. `numImages` from the
  Electron UI) that don't belong in a generation-engine package.

The bridge file is small (~200 lines) and entirely boilerplate, so
keeping it on the vMLX side is a net win.

## Request flow (end-to-end)

```
User clicks "Generate" in the SwiftUI ImageScreen
  ↓
ImageViewModel.generate() calls appState.engine.generateImage(...)
  ↓
Engine.generateImage (vMLX/FluxBackend.swift) lazy-creates a FluxEngine,
ensures the requested model is loaded, registers a FluxJobBridge for
live UI events, converts ImageGenSettings → vMLXFlux.ImageGenRequest
  ↓
vMLXFlux.FluxEngine.generate(request) streams ImageGenEvent back
  ↓
Engine bridges each event through FluxJobBridge.yield, ImageScreen
subscribes via Engine.imageGenStream(jobId:) — the UI's step counter,
elapsed timer, and partial preview render in real time
  ↓
Final .completed yields the output URL; Engine returns it from
generateImage; ImageScreen refreshes the gallery row
```

HTTP equivalent (for `/v1/images/generations` from external clients):

```
POST /v1/images/generations
  ↓
vMLXServer/Routes/OpenAIRoutes.createImage handler collects body,
builds ImageGenSettings, calls engine.generateImage, returns JSON
with the image URL (or multipart/base64 depending on the request)
```

## Directory layout

```
vmlx-flux/
├── Package.swift                 4 products: vMLXFlux + 3 sub-modules
├── README.md
├── ARCHITECTURE.md               ← this file
├── .gitignore
└── Sources/
    ├── vMLXFluxKit/              core types + math
    │   ├── FluxModel.swift         Capability protocols
    │   ├── Requests.swift          ImageGenRequest / ImageEditRequest / VideoGenRequest / events
    │   ├── ModelRegistry.swift     name → loader registry
    │   ├── FlowMatchScheduler.swift flow-matching Euler sampler
    │   ├── LatentSpace.swift       initial noise + patchify/unpatchify
    │   ├── MathOps.swift           timestep embedding, 2D RoPE, attention
    │   ├── WeightLoader.swift      safetensors shard → [String: MLXArray]
    │   ├── JangSupport.swift       bridge to vmlx-swift-lm JangLoader
    │   ├── VAE.swift               AutoencoderKL decoder (Flux family)
    │   ├── FluxDiT.swift           FluxDoubleStreamBlock / SingleStreamBlock / FluxDiTModel
    │   └── ImageIO.swift           @MainActor PNG writer
    ├── vMLXFluxModels/           concrete model impls
    │   ├── Exports.swift             Registers all models at launch
    │   ├── Flux1/Flux1.swift         Schnell, Dev, Kontext, Fill
    │   ├── Flux2Klein/Flux2Klein.swift  Klein + KleinEdit
    │   ├── ZImage/ZImage.swift       Z-Image Turbo (wired end-to-end)
    │   ├── QwenImage/QwenImage.swift Qwen-Image + QwenImageEdit
    │   ├── FIBO/FIBO.swift
    │   └── SeedVR2/SeedVR2.swift
    ├── vMLXFluxVideo/            video gen (Wan 2.x)
    │   ├── WanDiT.swift             Video transformer block + model
    │   ├── WanVAE3D.swift           3D causal VAE decoder
    │   ├── WanVideoIO.swift         @MainActor MP4 writer via AVAssetWriter
    │   └── WANModel.swift           Wan 2.1 / 2.2 end-to-end pipeline
    └── vMLXFlux/                 umbrella
        └── FluxEngine.swift         Top-level actor facade

Tests/vMLXFluxTests/
├── RegistryTests.swift           8 tests — model registration + lookup
└── ShapeTests.swift              8 tests — pure-Swift scheduler/config sanity
```

## Adding a new model (recipe)

1. Create `Sources/vMLXFluxModels/<Name>/<Name>.swift`
2. Conform to one of: `ImageGenerator`, `ImageEditor`, `ImageUpscaler`,
   `VideoGenerator`
3. Add a `static let _register: Void = { ModelRegistry.register(...) }()`
   idiom that declares your canonical name, display name, default steps,
   default guidance, and a loader closure
4. Instantiate real `FluxDiTModel` + `VAEDecoder` modules in your init
   using one of the `FluxDiTConfig.*` presets (or add a new preset to
   `FluxDiT.swift` if your hyperparameters differ)
5. Append your type to `vMLXFluxModels.registerAll()` in
   `vMLXFluxModels/Exports.swift` so the engine sees it at launch
6. If the model has a different checkpoint key layout than Flux1,
   document the translation in the weight-key-map docblock on
   `FluxDiTModel` (in `FluxDiT.swift`)

## Status matrix

| Concern | Flux1 Schnell | Flux1 Dev | Flux2 Klein | ZImage | QwenImage | FIBO | SeedVR2 | Wan 2.1 | Wan 2.2 |
|---|---|---|---|---|---|---|---|---|---|
| Registered | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Scheduler wired | ⚪ | ⚪ | ⚪ | ✅ | ⚪ | ⚪ | ⚪ | ✅ | ⚪ |
| DiT forward | ⚪ | ⚪ | ⚪ | ✅ | ⚪ | ⚪ | ⚪ | ✅ | ⚪ |
| VAE decode | ⚪ | ⚪ | ⚪ | ✅ | ⚪ | ⚪ | ⚪ | ✅ | ⚪ |
| PNG/MP4 out | ⚪ | ⚪ | ⚪ | ✅ | ⚪ | ⚪ | ⚪ | ✅ | ⚪ |
| Real weights | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| T5-XXL encoder | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | — | ❌ | ❌ |
| CLIP-L encoder | ❌ | ❌ | — | — | — | — | — | — | — |
| 3-axis RoPE | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | — | ❌ | ❌ |

✅ = wired & tested, ⚪ = same module exists but model file routes
to placeholder, ❌ = not yet started, — = N/A.

**Biggest remaining blocker for any real image output**: the
`.safetensors` → module tree key-map translator + T5-XXL / CLIP-L text
encoder ports. Without real text embeddings the transformer outputs
random noise regardless of prompt. With real weights + text encoders,
Z-Image (already fully wired end-to-end) becomes the first model to
produce real output.

## Known deferred items (documented in code)

| Item | Where | Impact |
|---|---|---|
| 3-axis RoPE for Flux | `FluxDoubleStreamBlock`, `FluxSingleStreamBlock` (`_ = rope // TODO`) | Positional information lost — garbage without it |
| `@ModuleInfo(key: ...)` decorators | `FluxDiTModel` | Real safetensors load won't map keys automatically |
| Wan self-attention O(N²) | `WanDiTBlock.callAsFunction` | Doesn't scale past 5-sec 720p clips. Real Wan uses windowed |
| MXTQ PRNG bit-validation | `NumPyPCG64.swift` (in vmlx-swift-lm) | MXTQ weights may decode wrong until validated vs NumPy |
| Runtime shape tests that touch MLX | `Tests/vMLXFluxTests/ShapeTests.swift` | Test binary lacks `default.metallib` — shape tests for forward passes run from the app, not `swift test` |

## Build / test

```
cd vmlx-flux
swift build                        # clean build across all 4 products
swift test                         # 16/16 passing (8 registry + 8 shape)
```

vMLX-side integration test: from the main vMLX repo,
```
cd vllm-mlx/swift
swift build                        # verifies vMLXEngine links vMLXFlux correctly
swift test                         # runs the full vMLX test suite including FluxBackend bridge
```
