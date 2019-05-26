#include "fixnum/warp_fixnum.cu"
#include "functions/modinv.cu"
#include "modnum/modnum_monty_redc.cu"
#include "util/cuda_wrap.h"
// #include "array/fixnum_array.h"

#include <stdio.h>
#include <cassert>
#include <string>
#include <vector>

using namespace cuFIXNUM;
using namespace std;

// each element in F_mnt_4_753 has length less than 
//  753 < 768 bits (96 bytes)
typedef warp_fixnum<128, u64_fixnum> fixnum;
typedef modnum_monty_redc<fixnum> modnum_monty;
typedef modnum_monty::modnum modnum;

// warp fixnum stores lower digit in lower lane, 
// which is little-endian 

__global__
void multiply_together_mod(fixnum *mnt4, fixnum *mnt6, int n,
    fixnum *pmnt4_inputs, fixnum *pmnt6_inputs,
    fixnum *pmnt4_output, fixnum *pmnt6_output) 
{
    int odd = n & 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_idx = idx / fixnum::layout::WIDTH;

    int laneIdx = fixnum::layout::laneIdx();
    modnum_monty fmnt4753(mnt4[laneIdx]);
    modnum_monty fmnt6753(mnt6[laneIdx]);

    modnum z4;
    modnum z6;
    if ((warp_idx != 0 || odd == 0) && (warp_idx*2 < n)){
        fmnt4753.mul(z4, pmnt4_inputs[warp_idx], pmnt4_inputs[warp_idx+n/2]);
        fmnt6753.mul(z6, pmnt6_inputs[warp_idx], pmnt6_inputs[warp_idx+n/2]);
        fixnum::set(pmnt4_inputs[warp_idx], fixnum::get(z4, laneIdx), laneIdx);
        fixnum::set(pmnt6_inputs[warp_idx], fixnum::get(z6, laneIdx), laneIdx);
    } 
    
    // modnum mx, my;
    // fixnum x(laneIdx == 0 ? 10 : 0);
    // fixnum y(laneIdx == 0 ? 11 : 0);

    // fmnt4753.to_modnum(mx, x);
    // fmnt4753.to_modnum(my, y);

    // modnum mz;
    // fmnt4753.mul(mz, mx, my);
    
    // fixnum z;
    // fmnt4753.from_modnum(z, mz);
}

vector<uint8_t> from_big_endian_hex(string hex)
{
    assert(hex.size() % 2 == 0);
    vector<uint8_t> ret(hex.size() / 2);

    // uint8_t *ret = new uint8_t[hex.size() / 2];
    auto ret_iter = ret.begin();
    for (auto iter = hex.rbegin(); iter != hex.rend(); iter++)
    {
        *ret_iter = (*iter - '0') + ((*(++iter) - '0') << 4);
        ret_iter++;
    }
    return ret;
}

template< typename T >
inline static uint8_t *as_byte_ptr(T *ptr) {
    return reinterpret_cast<uint8_t *>(ptr);
}

int main(int argc, char* argv[])
{
    assert(argc >= 4);
    // step 0. prepare common params: F_mnt4_753.q and F_mnt6_753.q
    fixnum *mnt4 = nullptr, *mnt6 = nullptr;
    vector<uint8_t> vmnt4, vmnt6;

    string mnt4_hex("01C4C62D92C41110229022EEE2CDADB7F997505B8FAFED5EB7E8F96C97D87307FDB925E8A0ED8D99D124D9A15AF79DB117E776F218059DB80F0DA5CB537E38685ACCE9767254A4638810719AC425F0E39D54522CDD119F5E9063DE245E8001");
    string mnt6_hex("01C4C62D92C41110229022EEE2CDADB7F997505B8FAFED5EB7E8F96C97D87307FDB925E8A0ED8D99D124D9A15AF79DB26C5C28C859A99B3EEBCA9429212636B9DFF97634993AA4D6C381BC3F0057974EA099170FA13A4FD90776E240000001");

    vmnt4 = from_big_endian_hex(mnt4_hex);
    vmnt6 = from_big_endian_hex(mnt6_hex);

    cuda_malloc_managed(&mnt4, fixnum::BYTES);
    cuda_malloc_managed(&mnt6, fixnum::BYTES);

    fixnum::from_bytes(as_byte_ptr(mnt4), vmnt4.data(), vmnt4.size());
    fixnum::from_bytes(as_byte_ptr(mnt6), vmnt6.data(), vmnt6.size());

    auto input = fopen(argv[2], "r");
    auto output = fopen(argv[3], "w");

    while (true) {
        // step 1. get inputs from file
        int n;
        size_t elts_read = fread((void *) &n, sizeof(size_t), 1, input);
        if (elts_read == 0) { break; }

        fixnum *pmnt4_inputs = nullptr;
        fixnum *pmnt6_inputs = nullptr;
        fixnum *pmnt4_output = nullptr;
        fixnum *pmnt6_output = nullptr;
        cuda_malloc_managed(&pmnt4_inputs, fixnum::BYTES * n);
        cuda_malloc_managed(&pmnt6_inputs, fixnum::BYTES * n);
        cuda_malloc_managed(&pmnt4_output, fixnum::BYTES);
        cuda_malloc_managed(&pmnt6_output, fixnum::BYTES);

        for (int i = 0; i < n; i++) {
            uint8_t tmp[12*8];
            fread((void *) tmp, 12*8, 1, input);
            fixnum::from_bytes(as_byte_ptr(&pmnt4_inputs[i]), tmp, sizeof(tmp));
        }
        for (int i = 0; i < n; i++) {
            uint8_t tmp[12*8];
            fread((void *) tmp, 12*8, 1, input);
            fixnum::from_bytes(as_byte_ptr(&pmnt6_inputs[i]), tmp, sizeof(tmp));
        }

        // step 2. run Kernel computation
        while (n > 1) {
            multiply_together_mod<<<(n+15)/16, 256>>>(mnt4, mnt6, n, pmnt4_inputs, pmnt6_inputs, pmnt4_output, pmnt6_output);
            cuda_device_synchronize();
            n >>= 1;
        }
        

        // step 3. write outputs into file
        {
            uint8_t tmp[12*8];
            fixnum::to_bytes(tmp, sizeof(tmp), as_byte_ptr(&pmnt4_inputs[0]));
            fwrite(tmp, sizeof(tmp), 1, output);
        }

        {
            uint8_t tmp[12*8];
            fixnum::to_bytes(tmp, sizeof(tmp), as_byte_ptr(&pmnt6_inputs[0]));
            fwrite(tmp, sizeof(tmp), 1, output);
        }

        cuda_free(pmnt4_inputs);
        cuda_free(pmnt6_inputs);
        cuda_free(pmnt4_output);
        cuda_free(pmnt6_output);
    }

    fclose(output);

    cuda_free(mnt4);
    cuda_free(mnt6);
    return 0;
}
