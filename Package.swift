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
// Ships as a Swift Package. vMLX imports it and wires Engine.generateImage /
// editImage / imageGenStream to FluxEngine.
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
        .library(name: "VMLXFlux", targets: ["VMLXFlux"]),
        // Split modules for fine-grained deps and test scoping.
        .library(name: "VMLXFluxKit", targets: ["VMLXFluxKit"]),
        .library(name: "VMLXFluxModels", targets: ["VMLXFluxModels"]),
        .library(name: "VMLXFluxVideo", targets: ["VMLXFluxVideo"]),
    ],
    dependencies: [
        // Apple Silicon tensor ops — same fork vmlx-swift-lm pins.
        .package(url: "https://github.com/osaurus-ai/mlx-swift", branch: "osaurus-0.31.3"),
        // Tokenizer + HuggingFace Hub loader.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.21"),
    ],
    targets: [
        // MARK: - VMLXFluxKit — core types + math + schedulers
        //
        // The "runtime" layer: diffusion schedulers (Euler, FlowMatch,
        // DDIM, DPM-Solver++), CFG helper, latent blending, noise
        // generation, image ↔ latent round-trip (VAE), safety checker.
        // Model-agnostic — every model in VMLXFluxModels depends on this.
        .target(
            name: "VMLXFluxKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/VMLXFluxKit"
        ),

        // MARK: - VMLXFluxModels — concrete model implementations
        //
        // Each model lives in its own subdirectory so contributors can add
        // new variants without touching existing ones. The umbrella target
        // re-exports each sub-model's public API via `@_exported import`.
        .target(
            name: "VMLXFluxModels",
            dependencies: [
                "VMLXFluxKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/VMLXFluxModels"
        ),

        // MARK: - VMLXFluxVideo — video generation (WAN 2.1/2.2, future)
        //
        // Scaffolded but not implemented. Protocol surface + empty files
        // so vMLX can compile against it today and the video models slot
        // in later without breaking vMLX's wiring.
        .target(
            name: "VMLXFluxVideo",
            dependencies: [
                "VMLXFluxKit",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/VMLXFluxVideo"
        ),

        // MARK: - VMLXFlux — umbrella
        //
        // The ONE import that callers usually want: `import VMLXFlux`
        // gives you `FluxEngine`, `ImageGenRequest`, `ImageEditRequest`,
        // `VideoGenRequest`, the model registry, and progress streams.
        .target(
            name: "VMLXFlux",
            dependencies: [
                "VMLXFluxKit",
                "VMLXFluxModels",
                "VMLXFluxVideo",
            ],
            path: "Sources/VMLXFlux"
        ),

        // MARK: - Tests
        .testTarget(
            name: "VMLXFluxTests",
            dependencies: ["VMLXFlux"],
            path: "Tests/VMLXFluxTests"
        ),
    ]
)
