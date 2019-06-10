#include "fixnum/word_fixnum.cu"
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
// warp_fixnum can only support 512-bit, 1024-bit, and 2048-bit 
//  multi-precision integer.
typedef warp_fixnum<128, u64_fixnum> fixnum;
typedef modnum_monty_redc<fixnum> modnum_monty;
typedef modnum_monty::modnum modnum;

// warp fixnum stores lower digit in lower lane, 
// which is little-endian 

// inputs are in x*R1 (mod p) form where R1 = 2^768
// and input should be transformed into x*R2 (mod p) form
__global__
void preprocess_input(fixnum *mnt, int n, fixnum *inputs, fixnum *r1inv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_idx = idx / fixnum::layout::WIDTH;
    int laneIdx = fixnum::layout::laneIdx();

    modnum_monty mod(mnt[laneIdx]);

    // x*R1 * R1^(-1)*R2*R2*R2^(-1) (mod p)
    // = x*R2 (mod p)
    modnum r1invr2;
    modnum r1invr2r2;
    mod.to_modnum(r1invr2, r1inv[laneIdx]);
    mod.to_modnum(r1invr2r2, r1invr2);

    if (elem_idx < n) {
        modnum z;
        int offset = elem_idx*fixnum::layout::WIDTH + laneIdx;
	    //printf("[print input] fixnum %d: %lu\n", elem_idx, inputs[offset]);
        mod.mul(z, inputs[offset], r1invr2r2);
        inputs[offset] = z;
    }
}

__global__
void process_output(fixnum *mnt, fixnum *output, fixnum *r1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_idx = idx / fixnum::layout::WIDTH;
    int laneIdx = fixnum::layout::laneIdx();
    modnum_monty mod(mnt[laneIdx]);

    if (elem_idx == 0) {
        modnum z;
        mod.mul(z, output[laneIdx], r1[laneIdx]);
        output[laneIdx] = z;
	    //printf("[print output] lane %d: %lu\n", laneIdx, z);
    }
}

__global__
void multiply_together_mod(fixnum *mnt, int n, fixnum *inputs) 
{
    int odd = n & 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_idx = idx / fixnum::layout::WIDTH;
    int laneIdx = fixnum::layout::laneIdx();
    modnum_monty mod(mnt[laneIdx]);

    if ((elem_idx != 0 || odd == 0) && (elem_idx*2 < n)){
        // inputs[i] *= inputs[i+n/2];
        //int offset = elem_idx*fixnum::layout::WIDTH + laneIdx; // == idx;
        modnum z;
        int offset = idx;
        int next_offset = (n/2)*fixnum::layout::WIDTH + offset;
	    //printf("lane %d: %lu\n", laneIdx, inputs[offset]);
        mod.mul(z, inputs[offset], inputs[next_offset]);
        inputs[offset] = z;
    } 
}

__global__
void reduce_serial(fixnum *mnt, fixnum *prod, fixnum *input)
{
    int laneIdx = fixnum::layout::laneIdx();
    modnum_monty mod(mnt[laneIdx]);
    modnum z;
    mod.mul(z, prod[laneIdx], input[laneIdx]);
    prod[laneIdx] = z;
}


int hex_to_int(char hex)
{
    if (hex >= '0' && hex <= '9') 
        return hex - '0';
    else if (hex >= 'a' && hex <= 'f')
        return (hex-'a') + 10;
    else 
        return 0;
}

vector<uint8_t> from_big_endian_hex(string hex)
{
    vector<uint8_t> ret((hex.size() + 1) / 2);

    auto ret_iter = ret.begin();
    for (auto iter = hex.rbegin(); iter != hex.rend();iter++)
    {
        *ret_iter = hex_to_int(*iter);
        if (++iter == hex.rend()) 
            break;
        *ret_iter = hex_to_int(*iter) << 4;
        ret_iter++;
    }
    return ret;
}

template< typename T >
inline static uint8_t *as_byte_ptr(T *ptr) {
    return reinterpret_cast<uint8_t *>(ptr);
}

fixnum* fixnum_from_big_endian_hex(const char *c_hex)
{
    string hex(c_hex);
    fixnum *ret = nullptr;
    cuda_malloc_managed((void**) &ret, fixnum::BYTES);

    vector<uint8_t> v = from_big_endian_hex(hex);
    fixnum::from_bytes(as_byte_ptr(ret), v.data(), v.size());
    return ret;
}
int main(int argc, char* argv[])
{
    assert(argc >= 4);

    // step 0. prepare common params: F_mnt4_753.q and F_mnt6_753.q
    fixnum *mnt4 = fixnum_from_big_endian_hex("01c4c62d92c41110229022eee2cdadb7f997505b8fafed5eb7e8f96c97d87307fdb925e8a0ed8d99d124d9a15af79db117e776f218059db80f0da5cb537e38685acce9767254a4638810719ac425f0e39d54522cdd119f5e9063de245e8001");
    fixnum *mnt6 = fixnum_from_big_endian_hex("01c4c62d92c41110229022eee2cdadb7f997505b8fafed5eb7e8f96c97d87307fdb925e8a0ed8d99d124d9a15af79db26c5c28c859a99b3eebca9429212636b9dff97634993aa4d6c381bc3f0057974ea099170fa13a4fd90776e240000001");
    fixnum *r1inv_mnt4 = fixnum_from_big_endian_hex("014caaa5fdb8ac881a53812b241540f5dc6228d561dfe748b35daa0c2e19f5868da6ed9b2e9e509a0bb1af313cef75a7424e4de1234c79ee32c031de123cab3188b9ef1b53258fe2e5cebefc11b219acb1cf33b776445b9588038856054e5f");
    fixnum *r1inv_mnt6 = fixnum_from_big_endian_hex("011e5a97bb2e58e46705d56ce665bc736ea1dccfef54ba2fa4d8056bee5de51b6aacdd0f39ea3d372699b59f04124887cf93e64e9e4fe7c269b1d5e296eb3aeffe67ac22c4cf370041c413fa9c90f53ff36ce86b15f05e8d37cc1acec9ef55");
    fixnum *r1_mnt4 = fixnum_from_big_endian_hex("7b479ec8e24295455fb31ff9a1950fa47edb3865e88c4074c9cbfd8ca621598b4302d2f00a62320c3bb7133385591e0f4d8acf031d68ed269c942108976f79589819c788b60197c3e4a0cd14572e91cd31c65a03468698a8ecabd9dc6f42");
    fixnum *r1_mnt6 = fixnum_from_big_endian_hex("7b479ec8e24295455fb31ff9a1950fa47edb3865e88c4074c9cbfd8ca621598b4302d2f00a62320c3bb7133384989fbca908de0ccb62ab0c4ee6d3e6dad40f725caec549c0daa1ebd2d90c79e1794eb16817b589cea8b99680147fff6f42");

    auto input = fopen(argv[2], "r");
    auto output = fopen(argv[3], "w");

    while (true) {
        // step 1. get inputs from file
        size_t n, m;
        size_t elts_read = fread((void *) &n, sizeof(size_t), 1, input);
        if (elts_read == 0) { break; }

	m = n;
        fixnum *pmnt4_inputs;
        fixnum *pmnt6_inputs;
        // fixnum *pmnt4_output = nullptr;
        // fixnum *pmnt6_output = nullptr;
        cuda_malloc_managed((void**)&pmnt4_inputs, fixnum::BYTES * n);
        cuda_malloc_managed((void**)&pmnt6_inputs, fixnum::BYTES * n);
        // cuda_malloc_managed(&pmnt4_output, fixnum::BYTES);
        // cuda_malloc_managed(&pmnt6_output, fixnum::BYTES);

        for (int i = 0; i < n; i++) {
            uint8_t tmp[12*8];
            fread((void *) tmp, 12*8, 1, input);
            fixnum::from_bytes(as_byte_ptr(&pmnt4_inputs[i*fixnum::layout::WIDTH]), tmp, sizeof(tmp));
        }
        for (int i = 0; i < n; i++) {
            uint8_t tmp[12*8];
            fread((void *) tmp, 12*8, 1, input);
            fixnum::from_bytes(as_byte_ptr(&pmnt6_inputs[i*fixnum::layout::WIDTH]), tmp, sizeof(tmp));
        }

        // step 2. preprocess input
        preprocess_input<<<(n+15)/16, 256>>>(mnt4, n, pmnt4_inputs, r1inv_mnt4);
        cuda_device_synchronize();

        preprocess_input<<<(n+15)/16, 256>>>(mnt6, n, pmnt6_inputs, r1inv_mnt6);
        cuda_device_synchronize();
        
        // step 3. run Kernel computation
    #ifdef REDUCE_GPU
        printf("running in gpu accelerate mode\n");
        while (n > 1) {
            multiply_together_mod<<<(n+15)/16, 256>>>(mnt4, n, pmnt4_inputs);
            cuda_device_synchronize();

            multiply_together_mod<<<(n+15)/16, 256>>>(mnt6, n, pmnt6_inputs);
            cuda_device_synchronize();

	        if (n & 1 == 1) {
		        n = (n+1)/2;
	        }
	        else {
		        n = n/2;
	        }
        }
    #else
	    for (int i = 1; i < n; i++) {
		    reduce_serial<<<1, 16>>>(mnt4, &pmnt4_inputs[0], &pmnt4_inputs[i*fixnum::layout::WIDTH]);
		    cuda_device_synchronize();
        }
	    for (int i = 1; i < n; i++) {
		    reduce_serial<<<1, 16>>>(mnt6, &pmnt6_inputs[0], &pmnt6_inputs[i*fixnum::layout::WIDTH]);
		    cuda_device_synchronize();
	    }
    #endif
        process_output<<<1, 16>>>(mnt4, &pmnt4_inputs[0], r1_mnt4);
        cuda_device_synchronize();
        process_output<<<1, 16>>>(mnt6, &pmnt6_inputs[0], r1_mnt6);
        cuda_device_synchronize();

        // step 3. write outputs into file
        {
            uint8_t tmp[12*8];
            fixnum::to_bytes(tmp, sizeof(tmp), as_byte_ptr(&pmnt4_inputs[0]));
            fwrite(tmp, sizeof(tmp), 1, output);
	        // printf("n = %lu\n", m);
	        // for (int i = 0; i < 12; i++) {
	        //     printf("%lu\n", pmnt4_inputs[i]);
	        // }
        }

        {
            uint8_t tmp[12*8];
            fixnum::to_bytes(tmp, sizeof(tmp), as_byte_ptr(&pmnt6_inputs[0]));
            fwrite(tmp, sizeof(tmp), 1, output);
        }

        cuda_free(pmnt4_inputs);
        cuda_free(pmnt6_inputs);
        // cuda_free(pmnt4_output);
        // cuda_free(pmnt6_output);
    }

    fclose(output);

    cuda_free(mnt4);
    cuda_free(mnt6);
    cuda_free(r1_mnt4);
    cuda_free(r1_mnt6);
    cuda_free(r1inv_mnt4);
    cuda_free(r1inv_mnt6);

    return 0;
}

#if 0 
int main(int argc, char* argv[])
{
	auto input = fopen(argv[2], "r");

	while (true)
	{
		size_t n;
		size_t elts_read = fread((void*)&n, sizeof(n), 1, input);
		if (elts_read == 0) { break; }

		for (int i =0 ; i < n; i++) {
			uint8_t tmp[12*8];
			fread((void*)tmp, 12*8, 1, input);
			printf("%lu\n", ((uint64_t*)tmp)[0]);
		}
		for (int i =0 ; i < n; i++) {
			uint8_t tmp[12*8];
			fread((void*)tmp, 12*8, 1, input);
			printf("%lu\n", ((uint64_t*)tmp)[0]);
		}

	}
	fclose(input);
}
#endif
