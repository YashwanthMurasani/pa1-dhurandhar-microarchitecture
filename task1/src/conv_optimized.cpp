#include <immintrin.h>
#include <algorithm>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  
    
            
    for (int oy = 0; oy+1 < H; oy+=2) {
        for (int ox=0; ox < W; ox += 32) {
            __m256 v00 = _mm256_setzero_ps(); 
            __m256 v01 = _mm256_setzero_ps();
            __m256 v02 = _mm256_setzero_ps();          
            __m256 v03 = _mm256_setzero_ps(); 
            __m256 v10 = _mm256_setzero_ps(); 
            __m256 v11 = _mm256_setzero_ps(); 
            __m256 v12 = _mm256_setzero_ps(); 
            __m256 v13 = _mm256_setzero_ps(); 
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m256 v_ker = _mm256_set1_ps(ker[ky * K + kx]);
                    v00 = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]),v_ker,v00);
                    v01 = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx + 8)]),v_ker,v01);
                    v02 = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx + 16)]),v_ker,v02);
                    v03 = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx + 24)]),v_ker,v03);
                    v10 = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + ky + 1) * in_stride + (ox + kx)]),v_ker,v10);
                    v11 = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + ky + 1) * in_stride + (ox + kx + 8)]),v_ker,v11);
                    v12 = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + ky + 1) * in_stride + (ox + kx + 16)]),v_ker,v12);
                    v13 = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + ky + 1) * in_stride + (ox + kx + 24)]),v_ker,v13);
                }
            }
            _mm256_storeu_ps(&out[oy * W + ox], v00);
            _mm256_storeu_ps(&out[oy * W + ox + 8], v01);
            _mm256_storeu_ps(&out[oy * W + ox + 16], v02);
            _mm256_storeu_ps(&out[oy * W + ox + 24], v03);
            _mm256_storeu_ps(&out[(oy + 1) * W + ox], v10);
            _mm256_storeu_ps(&out[(oy + 1) * W + ox + 8], v11);
            _mm256_storeu_ps(&out[(oy + 1) * W + ox + 16], v12);
            _mm256_storeu_ps(&out[(oy + 1) * W + ox + 24], v13);
        }            
    }

    for (int oy = H - (H % 2); oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += 32) {
            __m256 v00 = _mm256_setzero_ps(); 
            __m256 v01 = _mm256_setzero_ps();
            __m256 v02 = _mm256_setzero_ps();          
            __m256 v03 = _mm256_setzero_ps(); 
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m256 v_ker = _mm256_set1_ps(ker[ky * K + kx]);
                    v00 = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]),v_ker,v00);
                    v01 = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx + 8)]),v_ker,v01);
                    v02 = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx + 16)]),v_ker,v02);
                    v03 = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx + 24)]),v_ker,v03);
                }
            }
            _mm256_storeu_ps(&out[oy * W + ox], v00);
            _mm256_storeu_ps(&out[oy * W + ox + 8], v01);
            _mm256_storeu_ps(&out[oy * W + ox + 16], v02);
            _mm256_storeu_ps(&out[oy * W + ox + 24], v03);
        }
    }
}