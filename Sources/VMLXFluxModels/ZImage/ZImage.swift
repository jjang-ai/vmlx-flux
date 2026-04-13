import Foundation
import VMLXFluxKit

// Z-Image-Turbo — single-encoder turbo model, ~2B params, 4-8 steps.
// Python source: `mflux.models.z_image.variants.z_image.ZImage`.

public final class ZImage: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "z-image-turbo",
            displayName: "Z-Image Turbo",
            kind: .imageGen,
            defaultSteps: 4,
            defaultGuidance: 0.0,
            supportsLoRA: false,
            loader: { path, quant in
                _ = ZImage._register
                return try ZImage(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?

    public init(modelPath: URL, quantize: Int?) throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
    }

    public func generate(_ request: ImageGenRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: FluxError.notImplemented(
                "ZImage.generate — port from mflux/models/z_image/variants/z_image.py"))
        }
    }
}
