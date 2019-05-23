#include <stdio.h>

#include "fixnum/internal/primitives.cu"
#include "fixnum/word_fixnum.cu"
#include "fixnum/warp_fixnum.cu"

using namespace cuFIXNUM::internal;
using namespace cuFIXNUM;

__global__ 
void test()
{
	//u32 a, b, c;
	//a = 15; 
	//b = 12;
	//add_cc(c, a, b);
	//printf("c = %d\n", c);

	u32_fixnum a(11), b(22), c(33);
	u32_fixnum d;

	u32_fixnum::addc_cc(d, a, b);

	if (u32_fixnum::cmp(d, c) == 0) {
		printf("u32_fixnum.addc_cc: ok\n");
	}
}

__global__
void warp_add()
{

}

int main()
{
	warp_fixnum<16, u32_fixnum> *pa;
	warp_fixnum<16, u32_fixnum> *pb;

	cudaMallocManaged(&pa, warp_fixnum<16, u32_fixnum>::BYTES);
	cudaMallocManaged(&pb, warp_fixnum<16, u32_fixnum>::BYTES);

	
	//test << <1, 1 >> > ();
	return 0;
}