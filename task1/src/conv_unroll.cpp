// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    const int p=K/2;
    const int in_stride=W+2*p;  // padded row stride

    for(int oy=0;oy<H;++oy){
        for(int ox=0;ox<W;ox+=8){
            float acc0=0.0f;
            float acc1=0.0f;
            float acc2=0.0f;
            float acc3=0.0f;
            float acc4=0.0f;
            float acc5=0.0f;
            float acc6=0.0f;
            float acc7=0.0f;
            for(int ky=0;ky<K;++ky){
                for(int kx=0;kx<K;++kx){
                    float temp=ker[ky*K+kx];
                    int index=(oy+ky)*in_stride+(ox+kx);
                    acc0+=in[index+0]*temp;
                    acc1+=in[index+1]*temp;
                    acc2+=in[index+2]*temp;
                    acc3+=in[index+3]*temp;
                    acc4+=in[index+4]*temp;
                    acc5+=in[index+5]*temp;
                    acc6+=in[index+6]*temp;
                    acc7+=in[index+7]*temp;
                }
            }
            out[oy*W+ox+0]=acc0;
            out[oy*W+ox+1]=acc1;
            out[oy*W+ox+2]=acc2;
            out[oy*W+ox+3]=acc3;
            out[oy*W+ox+4]=acc4;
            out[oy*W+ox+5]=acc5;
            out[oy*W+ox+6]=acc6;
            out[oy*W+ox+7]=acc7;
        }
    }
}