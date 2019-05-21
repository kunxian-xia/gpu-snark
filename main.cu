
#define N 256

__global__ void add(int *pa, int *pb, int *pc)
{
    int tid = blockIdx.x;
    if (tid < N) 
        c[tid] = a[tid] + b[tid];
}

int main()
{
    int M = 1 << 16; // 

    int ha[M], hb[M], hc[M];
    int *da, *db, *dc;

    ha[0] = hb[0] = hc[0] = 1;
    for (int i = 1; i < M; i++) {
        ha[i] = (2*ha[i-1] + 7) % 753;
        hb[i] = (3*hb[i-1] + 17) % 753;
        hc[i] = (5*hc[i-1] + 47) % 753;
    }

    cudaMalloc((void**) &da, N*sizeof(int));
    cudaMalloc((void**) &db, N*sizeof(int));
    cudaMalloc((void**) &dc, N*sizeof(int));

    for (int i = 0; i < M/N; i++) {
        cudaMemcpy(da, a+i*N, N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(db, b+i*N, N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dc, c+i*N, N*sizeof(int), cudaMemcpyHostToDevice);

        add<<<N, 1>>>(da, db, dc);

        cudaMemcpy(c+i*N, dc, N*sizeof(int), cudaMemcpyDeviceToHost);
        // for (int j = 0; j < N; )
    }
    // cudaMemcpy();
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}