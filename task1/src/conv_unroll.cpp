// conv_simd.cpp  STAGE 4: SIMD with SSE/m128 intrinsics
#include <immintrin.h>
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  

    for (int oy = 0; oy < H; ++oy) {
        // Step by 4 since __m128 processes 4 floats at a time
        for (int ox = 0; ox < W; ox += 4) {
            __m128 v = _mm_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m128 v_ker = _mm_set1_ps(ker[ky * K + kx]);
                    __m128 v_in = _mm_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    v = _mm_fmadd_ps(v_in, v_ker, v);
                }
            }
            _mm_storeu_ps(&out[oy * W + ox], v);
        }
    }
}