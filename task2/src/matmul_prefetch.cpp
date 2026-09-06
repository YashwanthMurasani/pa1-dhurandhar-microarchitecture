// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"
// Tunable parameters
constexpr int DIST = 16;          // Distance D (e.g., 16, 32, 64, 128)
#define HINT _MM_HINT_T2         // Cache level: _MM_HINT_T0 / T1 / T2 / NTA

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    for(int i=0;i<M;++i){
        const float* a=A+static_cast<long>(i)*lda;
        // _mm_prefetch(reinterpret_cast<const char*>(a+p+DIST), HINT);
        for(int j=0;j<N;++j){
            float acc=0.0f;
            const float* b=B+static_cast<long>(j)*ldb;
            for(int p=0;p<K;++p){
                // Prefetch cache lines ahead
                if((p&7)==0){
                    _mm_prefetch(reinterpret_cast<const char*>(a+p+DIST), HINT);
                    _mm_prefetch(reinterpret_cast<const char*>(b+p+DIST), HINT);
                }

                acc+=a[p]*b[p];
            }
            C[static_cast<long>(i)*ldc+j]=acc;
        }
    }
}
