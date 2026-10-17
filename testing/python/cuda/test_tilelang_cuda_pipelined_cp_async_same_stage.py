import torch

import tilelang
import tilelang.language as T
import tilelang.testing

N, D, BM, THREADS = 1024, 128, 64, 128


def _make_kernel(block_c, v_stage):
    trips = N // block_c

    @T.prim_func
    def main(
        Q: T.Tensor((N, D), T.bfloat16), K: T.Tensor((N, D), T.bfloat16), V: T.Tensor((N, D), T.bfloat16), O: T.Tensor((N, D), T.float32)
    ):
        with T.Kernel(N // BM, threads=THREADS) as bx:
            Qs = T.alloc_shared((BM, D), T.bfloat16)
            Ks = T.alloc_shared((block_c, D), T.bfloat16)
            Vs = T.alloc_shared((block_c, D), T.bfloat16)
            S = T.alloc_fragment((BM, block_c), T.float32)
            P = T.alloc_fragment((BM, block_c), T.bfloat16)
            acc = T.alloc_fragment((BM, D), T.float32)
            T.copy(Q[bx * BM : (bx + 1) * BM, 0:D], Qs, disable_tma=True)
            T.clear(acc)
            for t in T.Pipelined(trips, order=[0, 1, 2, 3, 4], stage=[0, 1, 1, v_stage, 1]):
                T.copy(K[t * block_c : (t + 1) * block_c, 0:D], Ks, disable_tma=True)
                T.gemm(Qs, Ks, S, transpose_B=True, clear_accum=True)
                T.copy(S, P)
                for i, j in T.Parallel(block_c, D):
                    Vs[i, j] = T.if_then_else(True, V[t * block_c + i, j], 0)
                T.gemm(P, Vs, acc)
            T.copy(acc, O[bx * BM : (bx + 1) * BM, 0:D])

    return main


@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_cp_async_copy_in_consumer_stage_is_waited_for():
    # The V copy is committed as its own cp.async group right before its consumer in the same
    # pipeline stage, so the wait in front of the consumer must not leave that group in
    # flight (#3526).
    block_c = 128
    kernel = tilelang.compile(_make_kernel(block_c, v_stage=1), target="cuda")
    generator = torch.Generator(device="cuda").manual_seed(1)
    q = (torch.randn(N, D, device="cuda", generator=generator) * 0.25).bfloat16()
    k = (torch.randn(N, D, device="cuda", generator=generator) * 0.25).bfloat16()
    v = torch.randn(N, D, device="cuda", generator=generator).bfloat16()
    ref = torch.zeros(N, D, dtype=torch.float64, device="cuda")
    for t in range(N // block_c):
        rows = slice(t * block_c, (t + 1) * block_c)
        p = (q.float() @ k[rows].float().T).bfloat16().double()
        ref += p @ v[rows].double()
    tol = 1e-1 + 1e-2 * ref.abs()
    for _ in range(8):
        out = torch.full((N, D), float("nan"), device="cuda")
        kernel(q, k, v, out)
        torch.cuda.synchronize()
        err = (out.double() - ref).abs().nan_to_num(float("inf"))
        assert int((err > tol).sum()) == 0
