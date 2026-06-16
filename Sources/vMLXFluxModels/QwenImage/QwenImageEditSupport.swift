import Foundation
import vMLXFluxKit

public struct QwenImageEditPreprocessPlan: Sendable {
    public let outputWidth: Int
    public let outputHeight: Int
    public let vlWidth: Int
    public let vlHeight: Int
    public let vaeWidth: Int
    public let vaeHeight: Int
    public let conditioningPatchRows: Int
    public let conditioningPatchColumns: Int
    public let steps: Int
    public let guidance: Float

    public init(
        sourceImage: URL,
        requestedWidth: Int?,
        requestedHeight: Int?,
        steps: Int,
        guidance: Float
    ) throws {
        let dimensions = try ImageIO.dimensions(of: sourceImage)
        try self.init(
            sourceWidth: dimensions.width,
            sourceHeight: dimensions.height,
            requestedWidth: requestedWidth,
            requestedHeight: requestedHeight,
            steps: steps,
            guidance: guidance)
    }

    public init(
        sourceWidth: Int,
        sourceHeight: Int,
        requestedWidth: Int?,
        requestedHeight: Int?,
        steps: Int,
        guidance: Float
    ) throws {
        guard sourceWidth > 0, sourceHeight > 0 else {
            throw FluxError.invalidRequest("Qwen edit source image dimensions must be positive")
        }
        guard steps > 0 else {
            throw FluxError.invalidRequest("Qwen edit steps must be greater than zero")
        }
        guard guidance.isFinite else {
            throw FluxError.invalidRequest("Qwen edit guidance must be finite")
        }

        let ratio = Double(sourceWidth) / Double(sourceHeight)
        let generated = Self.roundedAreaDimensions(area: 1024 * 1024, ratio: ratio)
        let outputWidth = Self.floorToMultiple(requestedWidth ?? generated.width, multiple: 16)
        let outputHeight = Self.floorToMultiple(requestedHeight ?? generated.height, multiple: 16)
        guard outputWidth > 0, outputHeight > 0 else {
            throw FluxError.invalidRequest("Qwen edit output dimensions must be at least 16 pixels")
        }

        let vl = Self.roundedAreaDimensions(area: 384 * 384, ratio: ratio)
        let vae = Self.roundedAreaDimensions(area: 1024 * 1024, ratio: ratio)

        self.outputWidth = outputWidth
        self.outputHeight = outputHeight
        self.vlWidth = vl.width
        self.vlHeight = vl.height
        self.vaeWidth = vae.width
        self.vaeHeight = vae.height
        self.conditioningPatchRows = vae.height / 16
        self.conditioningPatchColumns = vae.width / 16
        self.steps = steps
        self.guidance = guidance
    }

    public static func imageIDs(height: Int, width: Int) throws -> [[Int]] {
        guard height > 0, width > 0, height % 16 == 0, width % 16 == 0 else {
            throw FluxError.invalidRequest("Qwen edit conditioning image dimensions must be positive multiples of 16")
        }
        let latentHeight = height / 16
        let latentWidth = width / 16
        var ids: [[Int]] = []
        ids.reserveCapacity(latentHeight * latentWidth)
        for row in 0 ..< latentHeight {
            for column in 0 ..< latentWidth {
                ids.append([1, row, column])
            }
        }
        return ids
    }

    private static func roundedAreaDimensions(area: Int, ratio: Double) -> (width: Int, height: Int) {
        let width = sqrt(Double(area) * ratio)
        let height = width / ratio
        return (
            roundToNearestEvenMultiple(width, multiple: 32),
            roundToNearestEvenMultiple(height, multiple: 32)
        )
    }

    private static func roundToNearestEvenMultiple(_ value: Double, multiple: Int) -> Int {
        Int((value / Double(multiple)).rounded(.toNearestOrEven)) * multiple
    }

    private static func floorToMultiple(_ value: Int, multiple: Int) -> Int {
        value / multiple * multiple
    }
}
