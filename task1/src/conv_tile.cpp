// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"
#include <algorithm>

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    int tile_H = 16;
    int tile_W = 128;

    for(int ty=0;ty<H;ty+=tile_H){
        int my = std::min(ty+tile_H,H);
        for(int tx=0;tx<W;tx+=tile_W){
            int mx = std::min(tx+tile_W,W);
            for (int oy = ty; oy < my; ++oy) {
                for (int ox = tx; ox < mx; ++ox) {
                    float acc = 0.0f;
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                        }
                    }
                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}