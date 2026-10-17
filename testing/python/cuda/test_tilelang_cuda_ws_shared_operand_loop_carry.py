import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _make_ws_gemm(K, num_stages):
    M = N = 128
    block_M = block_N = 128
    block_K = 64
    dtype, accum = T.float16, T.float32
    wgn = 2

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128 * wgn) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, "shared")
            B_shared_g0 = T.alloc_shared((block_K, block_N // wgn), dtype, "shared")
            B_shared_g1 = T.alloc_shared((block_K, block_N // wgn), dtype, "shared")
            C_local_g0 = T.alloc_fragment((block_M, block_N // wgn), accum)
            C_local_g1 = T.alloc_fragment((block_M, block_N // wgn), accum)
            with T.ws(1):
                T.clear(C_local_g1)
            with T.ws(0):
                T.clear(C_local_g0)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                with T.ws(1):
                    T.copy(B[ko * block_K, bx * block_N], B_shared_g1)
                    T.gemm(A_shared, B_shared_g1, C_local_g1)
                with T.ws(0):
                    T.copy(B[ko * block_K, bx * block_N + block_N // wgn], B_shared_g0)
                    T.gemm(A_shared, B_shared_g0, C_local_g0)
            with T.ws(1):
                T.copy(C_local_g1, C[by * block_M, bx * block_N])
            with T.ws(0):
                T.copy(C_local_g0, C[by * block_M, bx * block_N + block_N // wgn])

    return main


@pytest.mark.parametrize("K", [64, 128, 256])
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_ws_gemm_shared_a_is_not_overwritten_while_other_warpgroup_reads(K):
    # Both warpgroups read A_shared through WGMMA while every thread rewrites it in the next
    # K iteration. The barriers each warpgroup needs inside its own branch must not hide that
    # loop-carried write-after-read hazard (#2613).
    kernel = tilelang.compile(_make_ws_gemm(K, num_stages=0), out_idx=[2], target="cuda")
    generator = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(128, K, device="cuda", dtype=torch.float16, generator=generator)
    b = torch.randn(K, 128, device="cuda", dtype=torch.float16, generator=generator)
    ref = (a.float() @ b.float()).half()
    for _ in range(30):
        torch.testing.assert_close(kernel(a, b), ref, rtol=1e-2, atol=1e-1)
