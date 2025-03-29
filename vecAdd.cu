void vecAdd(float * A_h, float * B_h, float * c_h, int n){
    int size = n * sizeof(float);
    float * A_d, * B_d, * c_d;
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&c_d, size);
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
}