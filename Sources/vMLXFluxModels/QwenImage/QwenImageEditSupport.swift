import Foundation
@preconcurrency import MLX
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

public struct QwenImageEditVisionInput {
    public let pixelValues: MLXArray
    public let imageGridTHW: [Int]
    public let resizedWidth: Int
    public let resizedHeight: Int

    public var imageTokenCount: Int {
        imageGridTHW.reduce(1, *) / (QwenImageEditPreprocessor.mergeSize * QwenImageEditPreprocessor.mergeSize)
    }
}

public struct QwenImageEditVAEInput {
    public let tensor: MLXArray
}

public enum QwenImageEditPreprocessor {
    public static let patchSize = 14
    public static let temporalPatchSize = 2
    public static let mergeSize = 2

    public static func visionInput(
        sourceImage: URL,
        plan: QwenImageEditPreprocessPlan
    ) throws -> QwenImageEditVisionInput {
        let resized = try smartResize(height: plan.vlHeight, width: plan.vlWidth)
        let chw = try ImageIO.readRGBValues(
            sourceImage,
            width: resized.width,
            height: resized.height,
            normalization: .openAIClip)
        let gridT = 1
        let gridH = resized.height / patchSize
        let gridW = resized.width / patchSize
        guard resized.height % (patchSize * mergeSize) == 0,
              resized.width % (patchSize * mergeSize) == 0
        else {
            throw FluxError.invalidRequest("Qwen edit VL dimensions must be multiples of \(patchSize * mergeSize)")
        }

        let values = visionPatchValues(
            chw: chw,
            height: resized.height,
            width: resized.width,
            gridH: gridH,
            gridW: gridW)
        let pixelValues = MLXArray(
            values,
            [gridT * gridH * gridW, 3 * temporalPatchSize * patchSize * patchSize]
        ).asType(.float32)
        return QwenImageEditVisionInput(
            pixelValues: pixelValues,
            imageGridTHW: [gridT, gridH, gridW],
            resizedWidth: resized.width,
            resizedHeight: resized.height)
    }

    public static func vaeInput(
        sourceImage: URL,
        plan: QwenImageEditPreprocessPlan
    ) throws -> QwenImageEditVAEInput {
        let tensor = try ImageIO.readRGBTensor(
            sourceImage,
            width: plan.vaeWidth,
            height: plan.vaeHeight,
            normalization: .minusOneToOne)
        return QwenImageEditVAEInput(tensor: tensor)
    }

    public static func smartResize(
        height: Int,
        width: Int,
        factor: Int = patchSize * mergeSize,
        minPixels: Int = 56 * 56,
        maxPixels: Int = 28 * 28 * 1280
    ) throws -> (height: Int, width: Int) {
        guard height > 0, width > 0 else {
            throw FluxError.invalidRequest("Qwen edit image dimensions must be positive")
        }
        let aspect = Double(max(height, width)) / Double(min(height, width))
        guard aspect <= 200 else {
            throw FluxError.invalidRequest("Qwen edit image aspect ratio must be <= 200")
        }

        var resizedHeight = Int((Double(height) / Double(factor)).rounded(.toNearestOrEven)) * factor
        var resizedWidth = Int((Double(width) / Double(factor)).rounded(.toNearestOrEven)) * factor
        if resizedHeight * resizedWidth > maxPixels {
            let beta = sqrt(Double(height * width) / Double(maxPixels))
            resizedHeight = max(factor, Int(floor(Double(height) / beta / Double(factor))) * factor)
            resizedWidth = max(factor, Int(floor(Double(width) / beta / Double(factor))) * factor)
        } else if resizedHeight * resizedWidth < minPixels {
            let beta = sqrt(Double(minPixels) / Double(height * width))
            resizedHeight = Int(ceil(Double(height) * beta / Double(factor))) * factor
            resizedWidth = Int(ceil(Double(width) * beta / Double(factor))) * factor
        }
        return (resizedHeight, resizedWidth)
    }

    private static func visionPatchValues(
        chw: [Float],
        height: Int,
        width: Int,
        gridH: Int,
        gridW: Int
    ) -> [Float] {
        let plane = height * width
        let patchVectorLength = 3 * temporalPatchSize * patchSize * patchSize
        var values: [Float] = []
        values.reserveCapacity(gridH * gridW * patchVectorLength)

        for blockH in 0 ..< (gridH / mergeSize) {
            for blockW in 0 ..< (gridW / mergeSize) {
                for mergeH in 0 ..< mergeSize {
                    for mergeW in 0 ..< mergeSize {
                        let patchY = (blockH * mergeSize + mergeH) * patchSize
                        let patchX = (blockW * mergeSize + mergeW) * patchSize
                        for channel in 0 ..< 3 {
                            for _ in 0 ..< temporalPatchSize {
                                for y in 0 ..< patchSize {
                                    let rowOffset = (patchY + y) * width + patchX
                                    let channelOffset = channel * plane
                                    for x in 0 ..< patchSize {
                                        values.append(chw[channelOffset + rowOffset + x])
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return values
    }
}
