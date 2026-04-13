// swift-tools-version: 5.10
import PackageDescription

// vmlx-flux — native Apple Silicon image + video generation backend for vMLX.
//
// Scope:
//   - Image gen:   Flux1 Schnell/Dev, Flux2 Klein, ZImage Turbo, FIBO, QwenImage
//   - Image edit:  Flux1 Kontext, Flux1 Fill, Flux2 Klein Edit, QwenImage Edit
//   - Upscale:     SeedVR2
//   - Video (future): Apple WAN 2.1/2.2 via mlx-swift (scaffolded, not implemented)
//
// Ships as a Swift Package. vMLX imports it directly via
// `import vMLXFlux` and uses `vMLXFlux.FluxEngine` / `vMLXFluxModels.registerAll()` /
// `vMLXFluxVideo.registerAll()`. Module names use the lowercase-v
// `vMLX*` convention to match the rest of the vMLX Swift rewrite
// (vMLXEngine, vMLXServer, vMLXTheme, vMLXApp, vMLXCLI).
//
// Uses the same mlx-swift fork (osaurus-ai) as vmlx-swift-lm so the two
// packages share one MLX binary when co-installed.

let package = Package(
    name: "vmlx-flux",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        // Umbrella module — the single import most callers need.
        .library(name: "vMLXFlux", targets: ["vMLXFlux"]),
        // Split modules for fine-grained deps and test scoping.
        .library(name: "vMLXFluxKit", targets: ["vMLXFluxKit"]),
        .library(name: "vMLXFluxModels", targets: ["vMLXFluxModels"]),
        .library(name: "vMLXFluxVideo", targets: ["vMLXFluxVideo"]),
    ],
    dependencies: [
        // Apple Silicon tensor ops — same fork vmlx-swift-lm pins so both
        // packages share one MLX runtime when co-installed in vMLX.
        .package(url: "https://github.com/osaurus-ai/mlx-swift", branch: "osaurus-0.31.3"),
        // Sibling package — reused for JangConfig / JangLoader / TQDiskSerializer
        // so we don't re-port 533 lines of JANG v2 parsing.
        .package(path: "../vmlx-swift-lm"),
        // Tokenizer + HuggingFace Hub loader.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.21"),
    ],
    targets: [
        // MARK: - vMLXFluxKit — core types + math + schedulers
        //
        // The "runtime" layer: diffusion schedulers (Euler, FlowMatch,
        // DDIM, DPM-Solver++), CFG helper, latent blending, noise
        // generation, image ↔ latent round-trip (VAE), safety checker.
        // Model-agnostic — every model in vMLXFluxModels depends on this.
        .target(
            name: "vMLXFluxKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "vmlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/vMLXFluxKit"
        ),

        // MARK: - vMLXFluxModels — concrete model implementations
        //
        // Each model lives in its own subdirectory so contributors can add
        // new variants without touching existing ones. The umbrella target
        // re-exports each sub-model's public API via `@_exported import`.
        .target(
            name: "vMLXFluxModels",
            dependencies: [
                "vMLXFluxKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/vMLXFluxModels"
        ),

        // MARK: - vMLXFluxVideo — video generation (WAN 2.1/2.2, future)
        //
        // Scaffolded but not implemented. Protocol surface + empty files
        // so vMLX can compile against it today and the video models slot
        // in later without breaking vMLX's wiring.
        .target(
            name: "vMLXFluxVideo",
            dependencies: [
                "vMLXFluxKit",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/vMLXFluxVideo"
        ),

        // MARK: - vMLXFlux — umbrella
        //
        // The ONE import that callers usually want: `import vMLXFlux`
        // gives you `FluxEngine`, `ImageGenRequest`, `ImageEditRequest`,
        // `VideoGenRequest`, the model registry, and progress streams.
        .target(
            name: "vMLXFlux",
            dependencies: [
                "vMLXFluxKit",
                "vMLXFluxModels",
                "vMLXFluxVideo",
            ],
            path: "Sources/vMLXFlux"
        ),

        // MARK: - Tests
        .testTarget(
            name: "vMLXFluxTests",
            dependencies: ["vMLXFlux"],
            path: "Tests/vMLXFluxTests"
        ),
    ]
)
