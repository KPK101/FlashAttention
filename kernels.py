"""
#define SHARED_SIZE
__global__ void attention(float *x, float *xe, float *Wk, float *Wv, float *Wq, int d){
    /*
    Iter 0 of attention block
    Input arguments are weights for key, value, query, input vector
    Output is an embedding -  same shape as input
    */

    int tx = threadIdx.x;
    
    __shared__ float ashared[];
    int phases = d/SHARED_SIZE;
    float *
    for(int p=0; p<phases; p++){
        // compute product for each phase
        for(int i =0; i<SHARED_SIZE; i++){
            ashared[tx*SHARED_SIZE + i] = a[tx*SHARED_SIZE + i];
        }    
        
    }
    
    
            
}

__device__ float vecdot(float *asm, float *x){
    
    float result = 0;
    for(int i=0; i<SHARED_MEMORY; i++){
        result += a[i]*b[i];
    }
    return result;
    
}
"""