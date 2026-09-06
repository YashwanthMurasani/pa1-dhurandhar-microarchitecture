// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    for(int i=0;i<M;++i){
        const float* a=A+static_cast<long>(i)*lda;
        for(int j=0;j<N;++j){
            float acc=0.0f;
            const float* b=B+static_cast<long>(j)*ldb;
            int p=0;

            __m256 v_acc=_mm256_setzero_ps();
            for(;p+7<K;p+=8){
                __m256 va=_mm256_loadu_ps(a+p);
                __m256 vb=_mm256_loadu_ps(b+p);
                v_acc=_mm256_fmadd_ps(va,vb,v_acc);
            }
            __m128 lo = _mm256_castps256_ps128(v_acc);
            __m128 hi = _mm256_extractf128_ps(v_acc, 1);
            __m128 v = _mm_add_ps(lo, hi);
            v = _mm_hadd_ps(v, v);
            v = _mm_hadd_ps(v, v);
            acc += _mm_cvtss_f32(v);

            // __m128 v_acc=_mm_setzero_ps();
            // for(;p+3<K;p+=4){
            //     __m128 va=_mm_loadu_ps(a+p);
            //     __m128 vb=_mm_loadu_ps(b+p);
            //     v_acc=_mm_fmadd_ps(va,vb,v_acc);
            // }
            // v_acc = _mm_hadd_ps(v_acc, v_acc);
            // v_acc = _mm_hadd_ps(v_acc, v_acc);
            // acc += _mm_cvtss_f32(v_acc);

            for(;p<K;p++) acc+=a[p]*b[p];
            C[static_cast<long>(i)*ldc+j]=acc;
        }
    }
}
