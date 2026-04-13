import Foundation
import MLX
import MLXRandom

// MARK: - LatentSpace
//
// Utilities for constructing initial noise latents at the right shape
// for each model family. Flux and Qwen use a (B, H/8 × W/8, C=16)
// patchified layout; Z-Image uses (B, C=4, H/8, W/8). Both are 8x spatial
// downsamples of the target image resolution.
//
// Keeping these in VMLXFluxKit so every model ports against the same
// shape conventions.

public enum LatentLayout: Sendable {
    /// Flux-style: `(B, seq, channels)` where `seq = (H/8) × (W/8)` and
    /// channels is typically 16. Used by Flux1 / Flux2 / FIBO / Qwen.
    case fluxPatchified(channels: Int)
    /// Z-Image / legacy SDXL: `(B, C, H/8, W/8)`.
    case spatial(channels: Int)
}

public enum LatentSpace {

    /// Allocate an initial noise latent at the right shape for the model
    /// family. The `seed` parameter is threaded through `MLXRandom` so
    /// generations are reproducible when the user sets a seed.
    ///
    /// - Parameters:
    ///   - width: target image width in pixels (will be divided by 8)
    ///   - height: target image height in pixels
    ///   - layout: family-specific latent layout
    ///   - batchSize: how many images to generate in one forward pass
    ///   - seed: optional RNG seed (nil = system entropy)
    public static func initialNoise(
        width: Int,
        height: Int,
        layout: LatentLayout,
        batchSize: Int = 1,
        seed: UInt64? = nil
    ) -> MLXArray {
        // Seed the global MLX random generator. MLXRandom has a per-call
        // key parameter too; for simplicity we set-then-allocate.
        if let s = seed {
            MLXRandom.seed(s)
        }

        let latentH = height / 8
        let latentW = width / 8

        switch layout {
        case .fluxPatchified(let channels):
            let seqLen = latentH * latentW
            return MLXRandom.normal([batchSize, seqLen, channels])

        case .spatial(let channels):
            return MLXRandom.normal([batchSize, channels, latentH, latentW])
        }
    }

    /// Convert a patchified `(B, seq, C)` latent into a spatial
    /// `(B, C, H/8, W/8)` layout. Used before the VAE decode step.
    public static func unpatchify(
        _ latent: MLXArray,
        height: Int,
        width: Int
    ) -> MLXArray {
        let latentH = height / 8
        let latentW = width / 8
        let b = latent.dim(0)
        let channels = latent.dim(2)
        // (B, H*W, C) → (B, H, W, C) → (B, C, H, W)
        let reshaped = latent.reshaped([b, latentH, latentW, channels])
        return reshaped.transposed(0, 3, 1, 2)
    }

    /// The inverse of `unpatchify`. Used when the scheduler hands us a
    /// spatial tensor and the transformer wants a sequence.
    public static func patchify(_ latent: MLXArray) -> MLXArray {
        let b = latent.dim(0)
        let c = latent.dim(1)
        let h = latent.dim(2)
        let w = latent.dim(3)
        // (B, C, H, W) → (B, H, W, C) → (B, H*W, C)
        return latent.transposed(0, 2, 3, 1).reshaped([b, h * w, c])
    }
}
