import Foundation
import VMLXFluxKit

// MARK: - WAN (Wan 2.1 / Wan 2.2) — Apple Silicon video generation
//
// FUTURE / SCAFFOLDED. Apple research group has an open-source Wan 2.x
// video model family. When a Swift port lands (either in mlx-swift
// examples or as a community package), the model goes here and slots
// into `FluxEngine` via the `VideoGenerator` protocol.
//
// Design constraints for the future port:
//
//   - 121-frame generation at 1280×720, 24 fps → ~5-second clips
//   - Flow-matching scheduler (same family as Flux), different time schedule
//   - 3D temporal attention + 2D spatial attention
//   - T5-XXL text encoder (shared with Flux) — can reuse
//   - 3D causal VAE for encode/decode of video latents
//   - Progressive preview: decode every N-th step to a mid-res PNG strip
//
// Leave this file stubbed but compiling so vMLX can depend on
// VMLXFluxVideo today without blocking on video model availability.

public final class WANModel: VideoGenerator, @unchecked Sendable {
    /// Wan 2.1 — the original 1.3B / 14B variants.
    public static let _registerWan21: Void = {
        ModelRegistry.register(ModelEntry(
            name: "wan-2.1",
            displayName: "Wan 2.1",
            kind: .videoGen,
            defaultSteps: 50,
            defaultGuidance: 5.0,
            loader: { path, quant in
                _ = WANModel._registerWan21
                return try WANModel(modelPath: path, quantize: quant, version: .wan21)
            }
        ))
    }()

    /// Wan 2.2 — second generation with higher resolution defaults.
    public static let _registerWan22: Void = {
        ModelRegistry.register(ModelEntry(
            name: "wan-2.2",
            displayName: "Wan 2.2",
            kind: .videoGen,
            defaultSteps: 50,
            defaultGuidance: 5.0,
            loader: { path, quant in
                _ = WANModel._registerWan22
                return try WANModel(modelPath: path, quantize: quant, version: .wan22)
            }
        ))
    }()

    public enum Version: Sendable { case wan21, wan22 }

    public let modelPath: URL
    public let quantize: Int?
    public let version: Version

    public init(modelPath: URL, quantize: Int?, version: Version) throws {
        self.modelPath = modelPath
        self.quantize = quantize
        self.version = version
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
    }

    public func generate(_ request: VideoGenRequest) -> AsyncThrowingStream<VideoGenEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: FluxError.notImplemented(
                "WANModel.generate — Wan 2.x Swift port not yet implemented. "
                + "Tracking: https://github.com/ml-explore/mlx-examples for upstream."
            ))
        }
    }
}

/// Force-register the video models. Call once at app launch.
public enum VMLXFluxVideo {
    public static func registerAll() {
        _ = WANModel._registerWan21
        _ = WANModel._registerWan22
    }
}
