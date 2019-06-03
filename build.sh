nvcc -g -DNDEBUG -ccbin g++ -std=c++11 -Xcompiler -Wall,-Wextra \
    --gpu-architecture=compute_75 --gpu-code=sm_75  \
    -I./cuda-fixnum-v1/src/ -lstdc++ -o main main.cu
