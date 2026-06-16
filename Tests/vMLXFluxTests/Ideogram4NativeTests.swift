import MLX
import XCTest
@testable import vMLXFluxModels

final class Ideogram4NativeTests: XCTestCase {
    func testPromptInputsMatchMFluxPacking() throws {
        let inputs = try Ideogram4PromptInputs(
            tokenIDs: [10, 11, 12],
            width: 256,
            height: 256)

        XCTAssertEqual(inputs.tokenIDs.shape, [1, 259])
        XCTAssertEqual(inputs.textPositionIDs.shape, [1, 259, 3])
        XCTAssertEqual(inputs.positionIDs.shape, [1, 259, 3])
        XCTAssertEqual(inputs.segmentIDs.shape, [1, 259])
        XCTAssertEqual(inputs.indicator.shape, [1, 259])
        XCTAssertEqual(inputs.maxTextTokens, 3)
        XCTAssertEqual(inputs.numImageTokens, 256)
        XCTAssertEqual(inputs.gridHeight, 16)
        XCTAssertEqual(inputs.gridWidth, 16)

        XCTAssertEqual(Array(inputs.indicator.asArray(Int32.self).prefix(6)), [3, 3, 3, 2, 2, 2])
        XCTAssertEqual(Array(inputs.segmentIDs.asArray(Int32.self).prefix(6)), [1, 1, 1, 1, 1, 1])
        XCTAssertEqual(Array(inputs.positionIDs[0, 0, 0...].asArray(Int32.self)), [0, 0, 0])
        XCTAssertEqual(Array(inputs.positionIDs[0, 3, 0...].asArray(Int32.self)), [65536, 65536, 65536])
        XCTAssertEqual(Array(inputs.positionIDs[0, 4, 0...].asArray(Int32.self)), [65536, 65536, 65537])
        XCTAssertEqual(Array(inputs.positionIDs[0, 19, 0...].asArray(Int32.self)), [65536, 65537, 65536])
    }

    func testSchedulerMatchesDefaultMFluxReferenceValues() throws {
        let scheduler = try Ideogram4Scheduler(steps: 4, width: 256, height: 256, mu: 0, std: 1.75)

        XCTAssertEqual(scheduler.tValues.count, 4)
        XCTAssertEqual(scheduler.sValues.count, 4)
        XCTAssertEqual(scheduler.guidanceValues, [7, 7, 7, 7])
        XCTAssertEqual(scheduler.tValues[0], 0.866863, accuracy: 0.000001)
        XCTAssertEqual(scheduler.tValues[1], 0.666667, accuracy: 0.000001)
        XCTAssertEqual(scheduler.tValues[2], 0.380551, accuracy: 0.000001)
        XCTAssertEqual(scheduler.tValues[3], 0.000123, accuracy: 0.000001)
        XCTAssertEqual(scheduler.sValues[0], 0.999447, accuracy: 0.000001)
        XCTAssertEqual(scheduler.sValues[3], 0.380551, accuracy: 0.000001)
    }

    func testSchedulerUsesMFluxDefaultGuidancePreset() throws {
        let scheduler = try Ideogram4Scheduler(steps: 20, width: 256, height: 256, mu: 0, std: 1.75, guidance: 7)

        XCTAssertEqual(Array(scheduler.guidanceValues.prefix(4)), [3, 3, 7, 7])
        XCTAssertEqual(Array(scheduler.guidanceValues.suffix(2)), [7, 7])
        XCTAssertEqual(scheduler.guidanceValues.count, 20)
    }

    func testRotaryRotateHalfMatchesMFluxReference() throws {
        let x = MLXArray((1 ... 8).map(Float.init), [1, 1, 2, 4])

        let rotated = Ideogram4Rotary.rotateHalf(x)

        XCTAssertEqual(rotated.shape, [1, 1, 2, 4])
        XCTAssertEqual(rotated.asArray(Float.self), [-3, -4, 1, 2, -7, -8, 5, 6])
    }

    func testLatentUnpackAppliesNormAndPatchLayout() throws {
        let packed = MLXArray.zeros([1, 4, 128], dtype: .float32)

        let unpacked = try Ideogram4Latents.unpack(packed, width: 32, height: 32)

        XCTAssertEqual(unpacked.shape, [1, 32, 4, 4])
        XCTAssertEqual(unpacked[0, 0, 0, 0].item(Float.self), 0.01984364, accuracy: 0.0001)
        XCTAssertEqual(unpacked[0, 31, 0, 0].item(Float.self), -0.03571246, accuracy: 0.0001)
        XCTAssertEqual(unpacked[0, 0, 0, 1].item(Float.self), 0.02583857, accuracy: 0.0001)
        XCTAssertEqual(unpacked[0, 0, 1, 0].item(Float.self), 0.01924137, accuracy: 0.0001)
    }

    func testGenerateSourceNoLongerUsesNotImplementedStub() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Sources/vMLXFluxModels/Ideogram4/Ideogram4.swift"))

        XCTAssertFalse(source.contains("Ideogram4.generate — port from mflux/models/ideogram4"))
        XCTAssertFalse(source.contains("FluxError.notImplemented("))
    }
}
