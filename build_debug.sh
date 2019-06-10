nvcc -g -DNDEBUG -DREDUCE_GPU -ccbin g++ -std=c++11 -Xcompiler -Wall,-Wextra \
    --gpu-architecture=compute_35 --gpu-code=sm_35  \
    -I./cuda-fixnum-v1/src/ -lstdc++ -o main main.cu
