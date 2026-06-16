import Cmlx
@preconcurrency import MLX

func mfluxFromFP8(
    _ x: MLXArray,
    dtype: DType = .bfloat16,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_from_fp8(&result, x.ctx, dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}
