import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _make_gemm_kernel(dtype, accum_dtype):
    M = N = K = 64

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), accum_dtype)):
        with T.Kernel(1, 1, threads=128):
            As = T.alloc_shared((M, K), dtype)
            Bs = T.alloc_shared((K, N), dtype)
            Cl = T.alloc_fragment((M, N), accum_dtype)
            T.clear(Cl)
            T.copy(A, As)
            T.copy(B, Bs)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, C)

    return main


@tilelang.testing.requires_cuda
def test_gemm_bf16_accum_fp16_rejected_at_frontend():
    # Regression for bf16 x bf16 -> fp16: previously accepted by the frontend
    # and rejected by an nvcc static_assert; now must fail during lowering
    # with the native mma.sync diagnostic instead.
    with pytest.raises(tvm.error.InternalError, match=r"requires native mma\.sync lowering"):
        tilelang.compile(
            _make_gemm_kernel("bfloat16", "float16"),
            out_idx=[2],
            target={"kind": "cuda", "arch": "sm_89"},
        )


@pytest.mark.parametrize(
    "dtype,accum_dtype",
    [
        ("float16", "float16"),
        ("float16", "float32"),
        ("bfloat16", "float32"),
        (T.tfloat32, "float32"),
        ("float32", "float32"),
        ("float64", "float64"),
    ],
)
@tilelang.testing.requires_cuda
def test_gemm_mma_sync_supported_dtype_combo_lowers_on_ampere(dtype, accum_dtype):
    # Compile-only, pinned to sm_89 so the Ampere mma.sync whitelist (C++
    # AllowAmpereMMA) is actually exercised regardless of the CI runner's own
    # architecture. Regression coverage for #3004: tf32/f32/f64 must lower; the
    # old detection-based target="cuda" never ran this code on non-Ampere CI.
    tilelang.compile(
        _make_gemm_kernel(dtype, accum_dtype),
        out_idx=[2],
        target={"kind": "cuda", "arch": "sm_89"},
    )


@pytest.mark.parametrize(
    "dtype,accum_dtype",
    [
        ("float16", "float16"),
        ("float16", "float32"),
        ("bfloat16", "float32"),
    ],
)
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(8)
def test_gemm_mma_sync_supported_dtype_combo_runs(dtype, accum_dtype):
    import torch

    kernel = tilelang.compile(
        _make_gemm_kernel(dtype, accum_dtype),
        out_idx=[2],
        target="cuda",
    )
    torch.manual_seed(0)
    a = torch.randn(64, 64, device="cuda", dtype=getattr(torch, dtype))
    b = torch.randn(64, 64, device="cuda", dtype=getattr(torch, dtype))
    out = kernel(a, b)
    ref = (a.float() @ b.float()).to(getattr(torch, accum_dtype))
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def _make_fp8_gemm_kernel(b_k_major):
    M = N = 128
    K = 64
    dtype = T.float8_e4m3fn
    b_shape = (N, K) if b_k_major else (K, N)

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor(b_shape, dtype), C: T.Tensor((M, N), T.float16)):
        with T.Kernel(1, 1, threads=128):
            As = T.alloc_shared((M, K), dtype)
            Bs = T.alloc_shared(b_shape, dtype)
            Cl = T.alloc_fragment((M, N), T.float32)
            T.clear(Cl)
            T.copy(A, As)
            T.copy(B, Bs)
            T.gemm(As, Bs, Cl, transpose_B=b_k_major)
            T.copy(Cl, C)

    return main


@pytest.mark.parametrize("b_k_major,expect_warning", [(False, True), (True, False)])
@tilelang.testing.requires_cuda
def test_gemm_fp8_hopper_warns_when_layout_forces_mma_sync(capfd, b_k_major, expect_warning):
    # Regression for #3223: fp8 WGMMA needs K-major operands. A non-K-major B used
    # to drop to mma.sync with no diagnostic; K-major must stay silent.
    # Lower only (no nvcc, no kernel cache) so the warning fires on every run.
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    with target:
        tilelang.lower(_make_fp8_gemm_kernel(b_k_major), target=target)
    err = capfd.readouterr().err
    assert ("falls back from WGMMA to mma.sync" in err) == expect_warning
