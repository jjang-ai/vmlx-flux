# vmlx-flux

Native Apple Silicon image + video generation backend for **vMLX**.

Ships as a Swift Package. Shares the same [osaurus-ai/mlx-swift](https://github.com/osaurus-ai/mlx-swift) fork as [vmlx-swift-lm](https://github.com/jjang-ai/vmlx-swift-lm), so when co-installed in vMLX both packages use one MLX runtime.

## Models

### Image generation (text → image)
| Canonical name | Display | Backend |
|---|---|---|
| `flux1-schnell` | FLUX.1 Schnell | Flux1 (Black Forest Labs) |
| `flux1-dev` | FLUX.1 Dev | Flux1 |
| `flux2-klein` | FLUX.2 Klein | Flux2 |
| `z-image-turbo` | Z-Image Turbo | ZImage |
| `qwen-image` | Qwen-Image | QwenImage |
| `fibo` | FIBO | FIBO |

### Image edit (image + prompt → image)
| Canonical name | Display | Edit mode |
|---|---|---|
| `flux1-kontext` | FLUX.1 Kontext | prompt-only (no mask) |
| `flux1-fill` | FLUX.1 Fill | inpaint / outpaint (mask) |
| `flux2-klein-edit` | FLUX.2 Klein Edit | prompt + optional mask |
| `qwen-image-edit` | Qwen-Image-Edit | prompt + optional mask |

### Upscale (low-res → high-res)
| Canonical name | Display |
|---|---|
| `seedvr2` | SeedVR2 Upscaler |

### Video (future — scaffolded, not implemented)
| Canonical name | Display | Status |
|---|---|---|
| `wan-2.1` | Wan 2.1 | Scaffold only |
| `wan-2.2` | Wan 2.2 | Scaffold only |

## Install

Add to your `Package.swift` dependencies:

```swift
.package(url: "https://github.com/jjang-ai/vmlx-flux", branch: "main"),
```

Then depend on the product:

```swift
.target(
    name: "YourApp",
    dependencies: [
        .product(name: "VMLXFlux", package: "vmlx-flux"),
    ]
)
```

## Usage

```swift
import VMLXFlux

// One import gives you everything.
VMLXFluxModels.registerAll()   // registers all image + edit + upscale models
VMLXFluxVideo.registerAll()    // registers the video stubs (future)

let engine = FluxEngine()

// Load a model from a local weights dir — no silent HF downloads.
try await engine.load(
    name: "flux1-schnell",
    modelPath: URL(fileURLWithPath: "/Users/me/models/FLUX.1-schnell"),
    quantize: 8
)

// Stream a text-to-image generation.
let request = ImageGenRequest(
    prompt: "a cat riding a skateboard, photo",
    width: 1024, height: 1024,
    steps: 4,
    guidance: 0.0,
    outputDir: URL(fileURLWithPath: "/tmp/out")
)
for try await event in engine.generate(request) {
    switch event {
    case .step(let step, let total, _):
        print("step \(step)/\(total)")
    case .preview(_, let step):
        print("preview at step \(step)")
    case .completed(let url, _):
        print("saved to \(url)")
    case .failed(let msg, let hfAuth):
        print("failed: \(msg) (hfAuth=\(hfAuth))")
    case .cancelled:
        break
    }
}
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  VMLXFlux  (umbrella — one import for vMLX / apps)   │
│  ┌────────────────────────────────────────────────┐  │
│  │  FluxEngine  (actor — load / gen / edit / …)   │  │
│  └────────────────────────────────────────────────┘  │
│     │                      │                 │      │
│     ▼                      ▼                 ▼      │
│  VMLXFluxKit       VMLXFluxModels   VMLXFluxVideo    │
│   types, protocols,  Flux1,Flux2     WAN 2.1/2.2     │
│   schedulers,        ZImage, Qwen    (future)        │
│   VAE, requests,     FIBO, SeedVR2                   │
│   events                                             │
└──────────────────────────────────────────────────────┘
           │
           ▼
       mlx-swift (osaurus-ai/mlx-swift @ osaurus-0.31.3)
```

**Key design decisions**:

- **Protocol-based model dispatch** — `FluxModel` + `ImageGenerator` / `ImageEditor` / `ImageUpscaler` / `VideoGenerator` capability protocols. `FluxEngine` cast-dispatches based on `ModelKind` so adding a new model is additive.
- **Decentralized registration** — each model has a `static let _register: Void = { ModelRegistry.register(...) }()` so new variants land in their own file without touching central tables.
- **Streaming events** — every gen/edit/upscale returns `AsyncThrowingStream<ImageGenEvent, Error>`. Events: `.step`, `.preview` (optional partial decode), `.completed`, `.failed` (with `hfAuth` flag for 401/403), `.cancelled`.
- **No silent downloads** — `FluxEngine.load` REQUIRES a local path. The caller (vMLX `DownloadManager`) stages weights beforehand. Honors the vMLX rule: "NO silent downloads EVER".
- **Video scaffolded** — `VMLXFluxVideo` target exists and compiles today so vMLX can depend on it without blocking on Wan 2.x Swift availability.

## Status

**`z-image-turbo` is a working native pipeline** — `ZImageNative.swift` ports the
full model (Qwen-style text encoder + patchify/RoPE/adaLN DiT with noise/context
refiners + `AutoencoderKL` VAE + 4-bit mflux weight decode). Live-proven
2026-06-15 on the 4-bit mflux bundle: same-seed/same-prompt is byte-deterministic,
same-seed/different-prompt yields distinct coherent prompt-accurate images
(photographic apple vs watercolor mountain) at ~4 s per 512px/8-step image. Now
also vendored into `vmlx-swift` as in-tree targets so the whole vMLX stack shares
one MLX runtime — see `OSAURUS_VMLX_FLUX_INTEGRATION_SPEC.md` for the osaurus
wiring spec. `vmlxflux-probe` is the scan/load/generate verification CLI.

**The remaining models register but their generation bodies still throw
`FluxError.notImplemented`** (qwen-image(-edit), flux2-klein, flux1-*, seedvr2,
wan-2.x). Their ports need:

1. **FluxTransformer** (Dual-encoder Flux1 + single-encoder Flux2, DiT)
2. **T5-XXL text encoder** (shared across Flux + Qwen + Wan)
3. **CLIP-L text encoder** (Flux1 only)
4. **Autoencoder / VAE** (the Z-Image VAE decoder is a reusable reference)
5. **3D causal VAE** (video, future)
6. **Flow-matching scheduler** (Euler, sigmas 0→1 — done, shared)
7. **Safetensors weight loader** (4/8-bit — done for the mflux/JANG path)
8. **Tokenizer round-trip** via swift-transformers (done for Z-Image)

All of these plug into the existing `FluxEngine` + `ImageGenerator` protocol without API changes.

## License

TBD (likely MIT to match vmlx-swift-lm).
