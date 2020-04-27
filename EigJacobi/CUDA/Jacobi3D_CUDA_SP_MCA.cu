// Jacobi algorithm for eigen decomposition
// Eiegen decomp of a single 2D matrix 
// This function assumes that matrices have been input in row major format
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <chrono> //for timing
using namespace std::chrono;
#include<cuda_profiler_api.h> //Dont think i need this for debugging, just call executable with "nvprof a.out"

//#define USE_DOUBLE_PRECISION // to switch between single and double precision, can just comment this out and recompile
#ifdef USE_DOUBLE_PRECISION
typedef double floating;
#else
typedef float floating;
#endif

#define EPSILON 1e-4
#define THRESHOLD 1e-4 //1e-4 // Value determining convergence of algorithm 
#define MAX_BLOCK_SIZE 1024
#define MAX_SWEEPS 6//30
#define MAX_ITER 10000000
#define MULTIPLY_BLOCK_SIZE 64

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
    
void print_matrix(floating *A, int M, int N, bool console)
{
    
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (!console)
                fprintf(stderr, "%f ", A[i * N + j]);
            else
                printf("%f ", A[i * N + j]);
        }
        if (!console)
            fprintf(stderr, "\n");
        else
            printf("\n");
    }
    return;
}

void print_3Dmatrix(floating *A, int M, int N, int O, bool console)
{
    for (int k = 0; k < O; k++)        
    {        
        printf("k= %d \n",k);
        for (int i = 0; i < M; i++){
            for (int j = 0; j < N; j++){
                if (!console)
                    fprintf(stderr, "%f ", A[i * N + j + M*N*k]);
                else
                    printf("%f ", A[i * N + j + M*N*k]); 
            }
            if (!console)
                fprintf(stderr, "\n");
            else
                printf("\n");        
        }
    }
    return;
}


void s_initialize_identity(floating *I, int size)
{
    memset(I, 0, sizeof(floating)*size*size);
    for (int i = 0; i < size; i++)
        I[i * size + i] = 1.0;
}

void s_initialize_3Didentity(floating *I, int size, int depth)
{
    memset(I, 0, sizeof(floating)*size*size*depth);
    for (int k = 0; k < depth; k++)
    {
        for (int i = 0; i < size; i++)
        {
            I[i * size + i + size*size*k]=1.0; 
        }
    }
}

__global__ void kernel_s_initialize_3Didentity(floating *I, int size, int depth){
    int xID = blockDim.x*blockIdx.x + threadIdx.x;
    int yID = blockDim.y*blockIdx.y + threadIdx.y;
    int zID = blockDim.z*blockIdx.z + threadIdx.z;
    
    if (xID < size && yID < size && zID < depth){
        if (xID == yID)
            I[xID*size + yID + size*size*zID] = 1.0; 
    }
}

__device__ void chess_tourney_params(int P, int *row_pair, int iter)
{
    //NOTE: here, row_pair is thread-local
    int localID = threadIdx.x;
    int index1, index2;
    index1 = (localID + iter) % (P - 1);
    if (localID != 0)
    {
        index2 = (P - localID + iter - 1) % (P - 1);
    }
    else
    {
        index2 = P - 1;
    }
    row_pair[0] = min(index1, index2);
    row_pair[1] = max(index1, index2);
}

__global__ void kernel_compute_all_chess_params(int P, int Nk, int *device_IterBlockToElem)
{
    int blockID = blockIdx.x; 
    //each ONE of the P-1 blocks is responsible for computing chess-tourney parameters for ONE of the P-1 iterations
    int index = blockID*P + threadIdx.x*2; //using every thread to assign 2 adjacent values in array 'device_IterBlockToElem'
    //if (threadIdx.x < P/2){printf("%d\n",threadIdx.x);} //debugging
    assert(threadIdx.x < P/2);
    int *row_pair = (int *) malloc(sizeof(int)*2); //?? so every thread is allocating space on GPU 
    chess_tourney_params(P, row_pair, blockID);
    device_IterBlockToElem[index] = row_pair[0]; //|=(P-1)X(P/2*2)
    device_IterBlockToElem[index+1] = row_pair[1];
    
    /* This can be DELETED
    //the following loop should be replaced with a parallel implementation
    //printf("orig index %d \n", index);
    int stride = (P-1)*P/2*2; //number of elements in IterBlockToBlockToElem
    int inc_rowval = P*P;    
    for (int i=0; i<Nk; i++){         
        device_IterBlockToElem[index + i*stride] = row_pair[0] + i*inc_rowval; 
        device_IterBlockToElem[index +1 + i*stride] = row_pair[1] + i*inc_rowval;
        //printf("check index & iterblock val %d _ %d \n", index + i*stride, device_IterBlockToElem[index + i*stride]);
        //printf("check index & iterblock val %d _ %d \n", index + 1 + i*stride, device_IterBlockToElem[index + 1 + i*stride]);
    }
    */
    free(row_pair);
}

__global__ void kernel_compute_fmapping(int P, int *device_IterBlockToElem, int *device_fmapping)
{
    //*** this has not at all been optimized... Just scratch something together
    // assume that P-1 blocks have been launched, each with P/2 threads
    int localID = threadIdx.x;       
    extern __shared__ int stemp[];
    int *ginv = (int*) stemp;
    int *g = (int *) &ginv[P]; 
    g[localID*2]=device_IterBlockToElem[blockIdx.x*P + localID*2];
    g[localID*2+1]=device_IterBlockToElem[blockIdx.x*P + localID*2 +1];
    ginv[localID*2] = device_IterBlockToElem[blockIdx.x*P + localID*2+1]; //swapping every 2 adjacent elements
    ginv[localID*2+1] = device_IterBlockToElem[blockIdx.x*P + localID*2]; 
    
    /* //debugging
    if (localID==0 & blockIdx.x==0){
        printf("check ginv \n");
        for (int i=0; i<P; i++){
        printf("%d \n",g[i]); 
        }
    }
    */
    
    //really stupid but first pass at implementing 
    if (localID==0){
        for (int i=0; i<P; i++){
            for (int j=0; j<P/2; j++){          
                if (ginv[j*2]==i) {device_fmapping[blockIdx.x*P + i]= ginv[j*2 + 1];}
                if (g[j*2]==i) {device_fmapping[blockIdx.x*P + i] = g[j*2 + 1];}
            }  
        }        
    }
}


__global__ void kernel_compute_params(floating *device_A, int P, int iter, floating *device_sine, floating *device_cosine, int *device_IterBlockToElem)
{
    //builds the cos(theta) and sin(theta) matrices (each entry is the value for a corresponding k,l pair)
    //in 2D matrix case: *1 Block, P/2 threads: threadID t handles params for its alloted pair (for a particular device_iter)*/
    //in 3D matrix case: each block handles a separate matrix
    int localID = threadIdx.x;
    int globalID = blockIdx.x*blockDim.x + threadIdx.x;
    //if (localID < P/2){printf("%d\n",localID);} //debugging
    
    assert(localID < P / 2); //if assertion fails, not all threads terminate. note: p/2 will automatically evaluate to floor(p/2)        
    int k, l;
    floating elem, y, d, r, c, s; //,t
    k = device_IterBlockToElem[iter*P + localID*2]; //row
    l = device_IterBlockToElem[iter*P + localID*2+1]; //col
    //printf("blockID %d, k %d, l %d \n",blockIdx.x,k,l);
    elem = device_A[k * P + l + P*P*blockIdx.x];
    y = (device_A[l * P + l + P*P*blockIdx.x] - device_A[k * P + k + P*P*blockIdx.x]) * 0.5f;
    d = fabsf(y) + sqrtf(elem * elem + y * y);
    r = sqrtf(elem * elem + d * d);
    if (r < EPSILON){
        c = 1.0f;
        s = 0.0f;
    }
    else{
        //s = y / fabs(y) * elem / r; //t=y/fabs(y)*p*p/d; //original code        
        //s = elem/r; //this appears to be correct, but also requires more sweeps to converge than the line above
        if (y!=0.f){
            c = d / r;
            s = y / fabsf(y) * elem / r; 
            //s = elem/r;
            //s=sqrt(1-c*c);
        }
        else{
            c=1.0f/sqrtf(2.0f);
            s=1.0f/sqrtf(2.0f); //recommended to use different version
        }                
    }        
    device_cosine[k * P + l + P*P*blockIdx.x] = c;
    device_sine[k * P + l + P*P*blockIdx.x] = s;
    
    /* //debugging
    printf("tid_ %d, y %f \n",localID,y);
    printf("dev_s %f\n",device_sine[k*P+l]);
    printf("k_%d,l_%d \n",k,l);
    printf("dev_a1_%f, dev_a2_%f",device_A[l*P+l], device_A[k*P+k]); 
    */
}

__global__ void kernel_row_update(int iter, floating *device_A, floating *device_X, int P, floating *device_sine, floating *device_cosine, int *device_IterBlockToElem)
{
    
    int localID = threadIdx.x;
    int blockID = blockIdx.x;

    /*Based on blockID [total blocks=Nk*P/2], compute the corresponding two rows: p,q for device_iter*/
    /* shared memory version
    //For 3D case: every P/2 blocks
    __shared__ int row_pair[2];
    __shared__ floating params[2]; //[sin_, cos_]
    
    
    int kindex = blockID/(P/2); // this assumes that the kernel was launched with a gridsize= Nk*P/2   
    if (localID == 0)            //to minimize global memory access latency at the cost of divergence
    {          
        //printf("blockIdx %d, kindex %d \n ", blockID, kindex); 
        //printf("blockID %d, test %d \n",blockID, ((blockID)%(P/2))*2); 
        row_pair[0] = device_IterBlockToElem[iter*P + (blockID%(P/2)) * 2]; // sequence of row_pairs across blocks corresponding to 1 matrix should be the same for every other matrix
        row_pair[1] = device_IterBlockToElem[iter*P + (blockID%(P/2)) * 2 + 1];
        params[0] = device_sine[row_pair[0] * P + row_pair[1] + P*P*(kindex)]; // requires proper 3d indexing - note every P/2 blocks belong to same 2D matrix
        params[1] = device_cosine[row_pair[0] * P + row_pair[1] + P*P*(kindex)];
        //printf("blockID, %d row_pair %d \n",blockID,row_pair[0]); //row_pair should be a repeating sequence every P/2 blocks
        //printf("blockID, %d params[0] %f \n",blockID, params[0]); 
    }
    __syncthreads(); //all "P" threads in the block are synchronized and have access to row_pair(k,l) and params

    //CHECKPOINT: Can you reduce shared-memory bank conflicts here? 
    int k = row_pair[0], l = row_pair[1];
    floating sin_ = params[0], cos_ = params[1], elem_k=device_A[k*P+localID + P*P*kindex], elem_l=device_A[l * P + localID + P*P*kindex];
    */

    //For 3D case: every P/2 blocks
    int row_pair[2];
    floating params[2]; //[sin_, cos_]
         
    int kindex = blockID/(P/2); // this assumes that the kernel was launched with a gridsize= Nk*P/2   
        //printf("blockIdx %d, kindex %d \n ", blockID, kindex); 
        //printf("blockID %d, test %d \n",blockID, ((blockID)%(P/2))*2); 
        row_pair[0] = device_IterBlockToElem[iter*P + (blockID%(P/2)) * 2]; // sequence of row_pairs across blocks corresponding to 1 matrix should be the same for every other matrix
        row_pair[1] = device_IterBlockToElem[iter*P + (blockID%(P/2)) * 2 + 1];
        params[0] = device_sine[row_pair[0] * P + row_pair[1] + P*P*(kindex)]; // requires proper 3d indexing - note every P/2 blocks belong to same 2D matrix
        params[1] = device_cosine[row_pair[0] * P + row_pair[1] + P*P*(kindex)];
        //printf("blockID, %d row_pair %d \n",blockID,row_pair[0]); //row_pair should be a repeating sequence every P/2 blocks
        //printf("blockID, %d params[0] %f \n",blockID, params[0]); 
    
    //CHECKPOINT: Can you reduce shared-memory bank conflicts here? 
    int k = row_pair[0], l = row_pair[1];
    floating sin_ = params[0], cos_ = params[1], elem_k=device_A[k*P+localID + P*P*kindex], elem_l=device_A[l * P + localID + P*P*kindex];
    
    /*Concurrent modifications to all row pairs(k,l) [different blocks]*/
    /*Concurrent modifications to different-column elements of a row pair: ["P" threads of the block]*/

    /*X is col-major, i.e. write in X-transpose*/
    device_X[localID * P + k + P*P*kindex] = elem_k * cos_ - elem_l * sin_; 
    device_X[localID * P + l + P*P*kindex] = elem_k * sin_ + elem_l * cos_;
    
    //printf("devX_%f \n",device_X[localID*P +l]); //debugging
    //printf("cos_%f \n",cos_); //debugging    
    //printf("sin_%f \n", sin_); //debugging
    
}

__global__ void kernel_row_updateMCA(int iter, floating *device_A, floating *device_X, int P, floating *device_sine, floating *device_cosine, int *device_IterBlockToElem, int *device_fmapping)
{
    
    int localID = threadIdx.x;
    int blockID = blockIdx.x;

    /*Based on blockID [total blocks=Nk*P/2], compute the corresponding two rows: p,q for device_iter*/
    /* shared memory version
    //For 3D case: every P/2 blocks
    __shared__ int row_pair[2];
    __shared__ floating params[2]; //[sin_, cos_]
    
    
    int kindex = blockID/(P/2); // this assumes that the kernel was launched with a gridsize= Nk*P/2   
    if (localID == 0)            //to minimize global memory access latency at the cost of divergence
    {          
        //printf("blockIdx %d, kindex %d \n ", blockID, kindex); 
        //printf("blockID %d, test %d \n",blockID, ((blockID)%(P/2))*2); 
        row_pair[0] = device_IterBlockToElem[iter*P + (blockID%(P/2)) * 2]; // sequence of row_pairs across blocks corresponding to 1 matrix should be the same for every other matrix
        row_pair[1] = device_IterBlockToElem[iter*P + (blockID%(P/2)) * 2 + 1];
        params[0] = device_sine[row_pair[0] * P + row_pair[1] + P*P*(kindex)]; // requires proper 3d indexing - note every P/2 blocks belong to same 2D matrix
        params[1] = device_cosine[row_pair[0] * P + row_pair[1] + P*P*(kindex)];
        //printf("blockID, %d row_pair %d \n",blockID,row_pair[0]); //row_pair should be a repeating sequence every P/2 blocks
        //printf("blockID, %d params[0] %f \n",blockID, params[0]); 
    }
    __syncthreads(); //all "P" threads in the block are synchronized and have access to row_pair(k,l) and params

    //CHECKPOINT: Can you reduce shared-memory bank conflicts here? 
    int k = row_pair[0], l = row_pair[1];
    floating sin_ = params[0], cos_ = params[1], elem_k=device_A[k*P+localID + P*P*kindex], elem_l=device_A[l * P + localID + P*P*kindex];
    */

    //For 3D case: every P/2 blocks
    int row_pair[2];
    floating params[2]; //[sin_, cos_]
         
    int kindex = blockID/(P/2); // this assumes that the kernel was launched with a gridsize= Nk*P/2   
        //printf("blockIdx %d, kindex %d \n ", blockID, kindex); 
        //printf("blockID %d, test %d \n",blockID, ((blockID)%(P/2))*2); 
        row_pair[0] = device_IterBlockToElem[iter*P + (blockID%(P/2)) * 2]; // sequence of row_pairs across blocks corresponding to 1 matrix should be the same for every other matrix
        row_pair[1] = device_IterBlockToElem[iter*P + (blockID%(P/2)) * 2 + 1];
        params[0] = device_sine[row_pair[0] * P + row_pair[1] + P*P*(kindex)]; // requires proper 3d indexing - note every P/2 blocks belong to same 2D matrix
        params[1] = device_cosine[row_pair[0] * P + row_pair[1] + P*P*(kindex)];
        //printf("blockID, %d row_pair %d \n",blockID,row_pair[0]); //row_pair should be a repeating sequence every P/2 blocks
        //printf("blockID, %d params[0] %f \n",blockID, params[0]); 
    
    //CHECKPOINT: Can you reduce shared-memory bank conflicts here? 
    int k = row_pair[0], l = row_pair[1];
    floating sin_ = params[0], cos_ = params[1], elem_k=device_A[k*P+localID + P*P*kindex], elem_l=device_A[l * P + localID + P*P*kindex];
    
    /*Concurrent modifications to all row pairs(k,l) [different blocks]*/
    /*Concurrent modifications to different-column elements of a row pair: ["P" threads of the block]*/

    /*X is col-major, i.e. write in X-transpose*/
    //device_X[localID * P + k + P*P*kindex] = elem_k * cos_ - elem_l * sin_; 
    //device_X[localID * P + l + P*P*kindex] = elem_k * sin_ + elem_l * cos_;
    
    //printf("devX_%f \n",device_X[localID*P +l]); //debugging
    //printf("cos_%f \n",cos_); //debugging    
    //printf("sin_%f \n", sin_); //debugging
    
    ///////////////////TESTING THIS/////////////////////////////////
    ///////////////////// MCA attempt: X is stored in col-major form
    //for this, each block needs to know the k,l pair that every other block is working on for the current iteration       
    
    /*
    // Verbose and intractable version: (just shown for clarity in what's being done below without full kmatrix)
    floating ui, wi;
    int kset, lset;        
    int filocal = device_fmapping[iter*P + localID];    
    __shared__ floating kmat[16];    
    if (localID<P/2){
        kset = device_IterBlockToElem[iter*P + (localID%(P/2)) * 2]; 
        lset = device_IterBlockToElem[iter*P + (localID%(P/2)) * 2 + 1];
        
        kmat[kset*P + kset + P*P*kindex] = device_cosine[kset*P + lset + P*P*kindex];
        kmat[lset*P + lset + P*P*kindex] = device_cosine[kset*P + lset + P*P*kindex];
        kmat[kset*P + lset + P*P*kindex] = -device_sine[kset*P + lset + P*P*kindex];
        kmat[lset*P + kset + P*P*kindex] = device_sine[kset*P + lset + P*P*kindex];                
    }
    __syncthreads();
    ui = kmat[localID*P + localID + P*P*kindex];
    wi = kmat[localID*P + filocal + P*P*kindex];
            
    floating A_ip = elem_k; //read in col p, equivalent to read in row p    
    floating A_iq = elem_l;    
    floating A_fi_p = device_A[filocal*P + k + P*P*kindex]; //??? not coalesced, so should instead read, coalesced into shared mem?    
    floating A_fi_q = device_A[filocal*P + l + P*P*kindex];
    //form of writing in order to be coalesced
    device_X[k*P + localID + P*P*kindex] = A_ip * ui + A_fi_p * wi;
    device_X[l*P + localID + P*P*kindex] = A_iq * ui + A_fi_q * wi;
    
    printf("correct ui %f \n", ui);
    __syncthreads();    
    printf("new ui %f \n", ush[localID]);
    
    printf("correct wi %f \n", wi);
    __syncthreads();    
    printf("new wi %f \n", wsh[localID]);           
    
    if (blockID==1){        
        printf("indexes, X_%d%d, = A_%d%d, + A_%d%d \n", localID, k, localID, k, filocal, k);
        printf("\n");
        printf("indexes, X_%d%d, = A_%d%d, + A_%d%d \n", localID, l, localID, l, filocal, l);
    }
    */
    
    //version of the above without constructing full kmatrix    
    int kset, lset;        
    int filocal = device_fmapping[iter*P + localID];  
        
    extern __shared__ floating sharedarray[];
    floating *ush = (floating *) sharedarray;
    floating *wsh = (floating *) &ush[P];
    floating *row_k = (floating *) &wsh[P];
    floating *row_l = (floating *) &row_k[P];
    //__shared__ floating ush[128];
    //__shared__ floating wsh[128];
    //__shared__ floating row_k[128];
    //__shared__ floating row_l[128]; 
    
    row_k[localID] = device_A[k * P + localID + P*P*kindex];
    row_l[localID] = device_A[l * P + localID + P*P*kindex];
    
    if (localID<P/2){
        kset = device_IterBlockToElem[iter*P + (localID%(P/2)) * 2]; 
        lset = device_IterBlockToElem[iter*P + (localID%(P/2)) * 2 + 1];
           
        ush[kset] = device_cosine[kset*P + lset + P*P*kindex]; //need to triple check this
        ush[lset] = device_cosine[kset*P + lset + P*P*kindex];
        wsh[kset] = -device_sine[kset*P + lset + P*P*kindex]; 
        wsh[lset] = device_sine[kset*P + lset + P*P*kindex];    
    }
    __syncthreads();    
        
    //read the elements of A that are needed into shared memory            
    floating A_ip = elem_k; //read in col p, equivalent to read in row p    
    floating A_iq = elem_l;    
    //floating A_fi_p = device_A[filocal*P + k + P*P*kindex];     
    //floating A_fi_q = device_A[filocal*P + l + P*P*kindex];
    floating A_fi_p = row_k[filocal];
    floating A_fi_q = row_l[filocal];
    //form of writing in order to be coalesced
    device_X[k*P + localID + P*P*kindex] = A_ip * ush[localID] + A_fi_p * wsh[localID];
    device_X[l*P + localID + P*P*kindex] = A_iq * ush[localID] + A_fi_q * wsh[localID];    
}


__global__ void kernel_col_update(int iter, floating *device_A, floating *device_X, int P, 
    floating *device_eigenvectors, floating *device_sine, floating *device_cosine, int *device_IterBlockToElem)
{
    int localID = threadIdx.x;
    int blockID = blockIdx.x;

    /*Based on blockID [total blocks=P/2], compute the corresponding two cols: p,q for device_iter*/
    __shared__ int col_pair[2];
    __shared__ floating params[2]; //[sin_, cos_]
    
    int kindex = blockID/(P/2);
    if (localID == 0)            //to minimize global memory access latency at the cost of divergence
    {
        col_pair[0] = device_IterBlockToElem[iter*P + (blockID%(P/2)) * 2];
        col_pair[1] = device_IterBlockToElem[iter*P + (blockID%(P/2)) * 2 + 1];
        params[0] = device_sine[col_pair[0] * P + col_pair[1] + P*P*kindex];
        params[1] = device_cosine[col_pair[0] * P + col_pair[1] + P*P*kindex];
        //params[1] = sqrt(1-params[0]*params[0]); 
    }
    __syncthreads(); //all "P" threads in the block are synchronized and have access to row_pair(k,l) and params

    //CHECKPOINT: Can you reduce shared-memory bank conflicts here? Is this better than computing pair(p,q) all over again
    int k = col_pair[0], l = col_pair[1];
    floating sin_ = params[0], cos_ = params[1];

    /*Concurrent modifications to all row pairs(k,l) [different blocks]*/
    /*Concurrent modifications to different-column elements of a row pair: ["P" threads of the block]*/
    floating new_eigen_k, new_eigen_l;

    /*
    // col-wise access (inefficient)://    
    device_A[localID * P + k + P*P*kindex] = device_X[k * P + localID + P*P*kindex] * cos_ - device_X[l * P + localID + P*P*kindex] * sin_;
    device_A[localID * P + l + P*P*kindex] = device_X[k * P + localID + P*P*kindex] * sin_ + device_X[l * P + localID + P*P*kindex] * cos_;
    new_eigen_k = device_eigenvectors[localID * P + k + P*P*kindex]*cos_ - device_eigenvectors[localID*P+l + P*P*kindex]*sin_;
    new_eigen_l = device_eigenvectors[localID * P + k + P*P*kindex]*sin_ + device_eigenvectors[localID*P+l + P*P*kindex]*cos_;
    device_eigenvectors[localID * P + k + P*P*kindex] = new_eigen_k;
    device_eigenvectors[localID * P + l + P*P*kindex] = new_eigen_l;
    */
    
    //row-wise access (efficient)://
    int kp = k*P + localID + P*P*kindex, lp = l *P+localID + P*P*kindex;
    floating device_Xkp=device_X[kp], device_Xlp=device_X[lp];    
    device_A[kp] = device_Xkp * cos_ - device_Xlp * sin_;
    device_A[lp] = device_Xkp * sin_ + device_Xlp * cos_;
    
    //device_A[kp] = device_X[kp] * cos_ - device_X[lp] * sin_;
    //device_A[lp] = device_X[kp] * sin_ + device_X[lp] * cos_;
    new_eigen_k = device_eigenvectors[kp]*cos_ - device_eigenvectors[lp]*sin_;
    new_eigen_l = device_eigenvectors[kp]*sin_ + device_eigenvectors[lp]*cos_;
    device_eigenvectors[kp] = new_eigen_k;
    device_eigenvectors[lp] = new_eigen_l;        
}

floating compute_offset(floating *A, int P)
{
    floating sum = 0.0;
    for (int i = 0; i < P; i++)
    {
        for (int j = i + 1; j < P; j++)
        {
            sum += fabs(A[i * P + j]);
        }
    }
    return sum;
}

floating compute_offset3Dmatrix(floating *A, int P, int Nk)
{ //A is assumed to be a PxPxNk matrix
    floating sum = 0.0;
    for (int k = 0; k < Nk; k++){
    for (int i = 0; i < P; i++)
    {
        for (int j = i + 1; j < P; j++)
        {
            sum += fabs(A[i * P + j + P*P*k]);
        }
    }
    }
    return sum;
}

__global__ void kernel_compute_offset3Dmatrix_converge(floating *device_A, int P, int Nk, floating *offset, int *device_IterBlockToElem, bool *alg_converged){    
    //this kernel evaluates if the algorithm has converged for all matrices
    //assumptions about this kernel - launch as many blocks as there are 2D matrices to sum the off-diagonal elements of
    //???perhaps make the number of threads per block = min(#offdiagelements, max threads per block)
    int globalID = blockIdx.x*blockDim.x + threadIdx.x;
    int localID = threadIdx.x;
    int blockID = blockIdx.x;
    const int gridSize = blockDim.x*gridDim.x; 
    //device_IterBlockToElem already contains the i,j combinations of all off diagonal elements
    
    __shared__ floating ssum[MAX_BLOCK_SIZE];
    //extern __shared__ floating ssum[]; // Dynamic shared memory allocation (size specified on host)
    
    //device_IterBlockToElem contains consecutive pairs of row, column indices for the upperhalf, off-diagonal locations
    //recall for an element at row i, col j, depth k, index = i * P + j + P*P*k; 
    int kindex = blockID;
    int Noff = (P-1)*P/2; //number of off-diagonal elements (upperhalf) in a single 2D matrix
    int k, l;
    
    ssum[threadIdx.x] = 0.f;
    for (int index=localID; index<Noff; index += blockDim.x){ 
        k = device_IterBlockToElem[index*2]; //row
        l = device_IterBlockToElem[index*2 +1]; //col        
        ssum[localID] += fabs(device_A[k*P + l + P*P*kindex]); // for speed - consider a coalesced access pattern
        //parallel reduction on set of row,col inds
    }    
    __syncthreads();
    
    /* // debugging
    if (localID==0 && blockID==0){
    printf("check ssum:");
    for (int i=0; i<Noff; i++)
        printf("%f \n",ssum[i]);
    }
    */
    
    //step 2 - standard parallel reduce on shared var (after this, the "offset" of all matrices will still be distributed across different blocks)
     int j = threadIdx.x;
        for(int s = blockDim.x/2; s>0; s>>=1){
            if (j<s){
                ssum[j] += ssum[j+s];
            }
            __syncthreads();
        } 
    
    //step 3 - atomic sum to sum across blocks?
    //if (localID==0) atomicAdd(offset,ssum[0]);
    *offset = ssum[0];     
    
    //Is the following a bad idea:???  
    *alg_converged = true; // assume this is true until proven otherwise below
    if (localID==0){
        if (ssum[blockID]>THRESHOLD){
            *alg_converged = false; //i don't care how many matrices this is true for, only if it's true for one 
        }
    }     
}

__global__ void kernel_OneStepUpdate(int iter, floating *device_A, int P, floating *device_eigenvectors, floating *device_sine, floating *device_cosine, int *device_IterBlockToElem, int *device_fmapping)
{
    
    int localID = threadIdx.x;
    int blockID = blockIdx.x;

    /*Based on blockID [total blocks=Nk*P/2], compute the corresponding two rows: p,q for device_iter*/
    /* shared memory version
    //For 3D case: every P/2 blocks
    __shared__ int row_pair[2];
    __shared__ floating params[2]; //[sin_, cos_]
    
    
    int kindex = blockID/(P/2); // this assumes that the kernel was launched with a gridsize= Nk*P/2   
    if (localID == 0)            //to minimize global memory access latency at the cost of divergence
    {          
        //printf("blockIdx %d, kindex %d \n ", blockID, kindex); 
        //printf("blockID %d, test %d \n",blockID, ((blockID)%(P/2))*2); 
        row_pair[0] = device_IterBlockToElem[iter*P + (blockID%(P/2)) * 2]; // sequence of row_pairs across blocks corresponding to 1 matrix should be the same for every other matrix
        row_pair[1] = device_IterBlockToElem[iter*P + (blockID%(P/2)) * 2 + 1];
        params[0] = device_sine[row_pair[0] * P + row_pair[1] + P*P*(kindex)]; // requires proper 3d indexing - note every P/2 blocks belong to same 2D matrix
        params[1] = device_cosine[row_pair[0] * P + row_pair[1] + P*P*(kindex)];
        //printf("blockID, %d row_pair %d \n",blockID,row_pair[0]); //row_pair should be a repeating sequence every P/2 blocks
        //printf("blockID, %d params[0] %f \n",blockID, params[0]); 
    }
    __syncthreads(); //all "P" threads in the block are synchronized and have access to row_pair(k,l) and params

    //CHECKPOINT: Can you reduce shared-memory bank conflicts here? 
    int k = row_pair[0], l = row_pair[1];
    floating sin_ = params[0], cos_ = params[1], elem_k=device_A[k*P+localID + P*P*kindex], elem_l=device_A[l * P + localID + P*P*kindex];
    */

    //For 3D case: every P/2 blocks
    int row_pair[2];
    floating params[2]; //[sin_, cos_]
         
    int kindex = blockID/(P/2); // this assumes that the kernel was launched with a gridsize= Nk*P/2   
        //printf("blockIdx %d, kindex %d \n ", blockID, kindex); 
        //printf("blockID %d, test %d \n",blockID, ((blockID)%(P/2))*2); 
        row_pair[0] = device_IterBlockToElem[iter*P + (blockID%(P/2)) * 2]; // sequence of row_pairs across blocks corresponding to 1 matrix should be the same for every other matrix
        row_pair[1] = device_IterBlockToElem[iter*P + (blockID%(P/2)) * 2 + 1];
        params[0] = device_sine[row_pair[0] * P + row_pair[1] + P*P*(kindex)]; // requires proper 3d indexing - note every P/2 blocks belong to same 2D matrix
        params[1] = device_cosine[row_pair[0] * P + row_pair[1] + P*P*(kindex)];
        //printf("blockID, %d row_pair %d \n",blockID,row_pair[0]); //row_pair should be a repeating sequence every P/2 blocks
        //printf("blockID, %d params[0] %f \n",blockID, params[0]); 
    
    //CHECKPOINT: Can you reduce shared-memory bank conflicts here? 
    int k = row_pair[0], l = row_pair[1];
    floating sin_ = params[0], cos_ = params[1], elem_k=device_A[k*P+localID + P*P*kindex], elem_l=device_A[l * P + localID + P*P*kindex];
    
    /*Concurrent modifications to all row pairs(k,l) [different blocks]*/
    /*Concurrent modifications to different-column elements of a row pair: ["P" threads of the block]*/

    /*X is col-major, i.e. write in X-transpose*/
    //device_X[localID * P + k + P*P*kindex] = elem_k * cos_ - elem_l * sin_; 
    //device_X[localID * P + l + P*P*kindex] = elem_k * sin_ + elem_l * cos_;
    
    //printf("devX_%f \n",device_X[localID*P +l]); //debugging
    //printf("cos_%f \n",cos_); //debugging    
    //printf("sin_%f \n", sin_); //debugging
    
    ///////////////////TESTING THIS/////////////////////////////////
    ///////////////////// MCA attempt: X is stored in col-major form
    //for this, each block needs to know the k,l pair that every other block is working on for the current iteration       
    
    /*
    // Verbose and intractable version: (just shown for clarity in what's being done below without full kmatrix)
    floating ui, wi;
    int kset, lset;        
    int filocal = device_fmapping[iter*P + localID];    
    __shared__ floating kmat[16];    
    if (localID<P/2){
        kset = device_IterBlockToElem[iter*P + (localID%(P/2)) * 2]; 
        lset = device_IterBlockToElem[iter*P + (localID%(P/2)) * 2 + 1];
        
        kmat[kset*P + kset + P*P*kindex] = device_cosine[kset*P + lset + P*P*kindex];
        kmat[lset*P + lset + P*P*kindex] = device_cosine[kset*P + lset + P*P*kindex];
        kmat[kset*P + lset + P*P*kindex] = -device_sine[kset*P + lset + P*P*kindex];
        kmat[lset*P + kset + P*P*kindex] = device_sine[kset*P + lset + P*P*kindex];                
    }
    __syncthreads();
    ui = kmat[localID*P + localID + P*P*kindex];
    wi = kmat[localID*P + filocal + P*P*kindex];
            
    floating A_ip = elem_k; //read in col p, equivalent to read in row p    
    floating A_iq = elem_l;    
    floating A_fi_p = device_A[filocal*P + k + P*P*kindex]; //??? not coalesced, so should instead read, coalesced into shared mem?    
    floating A_fi_q = device_A[filocal*P + l + P*P*kindex];
    //form of writing in order to be coalesced
    device_X[k*P + localID + P*P*kindex] = A_ip * ui + A_fi_p * wi;
    device_X[l*P + localID + P*P*kindex] = A_iq * ui + A_fi_q * wi;
    
    printf("correct ui %f \n", ui);
    __syncthreads();    
    printf("new ui %f \n", ush[localID]);
    
    printf("correct wi %f \n", wi);
    __syncthreads();    
    printf("new wi %f \n", wsh[localID]);           
    
    if (blockID==1){        
        printf("indexes, X_%d%d, = A_%d%d, + A_%d%d \n", localID, k, localID, k, filocal, k);
        printf("\n");
        printf("indexes, X_%d%d, = A_%d%d, + A_%d%d \n", localID, l, localID, l, filocal, l);
    }
    */
    
    //version of the above without constructing full kmatrix    
    int kset, lset;        
    int filocal = device_fmapping[iter*P + localID];  
        
    extern __shared__ floating sharedarray[];
    floating *ush = (floating *) sharedarray;
    floating *wsh = (floating *) &ush[P];
    floating *row_k = (floating *) &wsh[P];
    floating *row_l = (floating *) &row_k[P];
    //__shared__ floating ush[128];
    //__shared__ floating wsh[128];
    //__shared__ floating row_k[128];
    //__shared__ floating row_l[128]; 
    
    row_k[localID] = device_A[k * P + localID + P*P*kindex];
    row_l[localID] = device_A[l * P + localID + P*P*kindex];
    
    if (localID<P/2){
        kset = device_IterBlockToElem[iter*P + (localID%(P/2)) * 2]; 
        lset = device_IterBlockToElem[iter*P + (localID%(P/2)) * 2 + 1];
           
        ush[kset] = device_cosine[kset*P + lset + P*P*kindex]; //need to triple check this
        ush[lset] = device_cosine[kset*P + lset + P*P*kindex];
        wsh[kset] = -device_sine[kset*P + lset + P*P*kindex]; 
        wsh[lset] = device_sine[kset*P + lset + P*P*kindex];    
    }
    __syncthreads();    
        
    //read the elements of A that are needed into shared memory            
    floating A_ip = elem_k; //read in col p, equivalent to read in row p    
    floating A_iq = elem_l;    
    //floating A_fi_p = device_A[filocal*P + k + P*P*kindex];     
    //floating A_fi_q = device_A[filocal*P + l + P*P*kindex];
    floating A_fi_p = row_k[filocal];
    floating A_fi_q = row_l[filocal];
    //form of writing in order to be coalesced
    //device_X[k*P + localID + P*P*kindex] = A_ip * ush[localID] + A_fi_p * wsh[localID];
    //device_X[l*P + localID + P*P*kindex] = A_iq * ush[localID] + A_fi_q * wsh[localID]; 
    
    device_A[k*P + localID + P*P*kindex] = (A_ip * ush[localID] + A_fi_p * wsh[localID])*cos_ - (A_iq * ush[localID] + A_fi_q * wsh[localID])*sin_;
    
    device_A[l*P + localID + P*P*kindex] = (A_ip * ush[localID] + A_fi_p * wsh[localID])*sin_ + (A_iq * ush[localID] + A_fi_q * wsh[localID])*cos_;
    
    floating new_eigen_k = device_eigenvectors[k*P + localID + P*P*kindex]*cos_ - device_eigenvectors[l*P + localID + P*P*kindex]*sin_;
    floating new_eigen_l = device_eigenvectors[k*P + localID + P*P*kindex]*sin_ + device_eigenvectors[l*P + localID + P*P*kindex]*cos_;
    device_eigenvectors[k*P + localID + P*P*kindex] = new_eigen_k;
    device_eigenvectors[l*P + localID + P*P*kindex] = new_eigen_l;
}

////////////////////Eigen value parallelization code
void EigJacobi(int P, int Nk, floating *A, floating *eigenvectors_T)
{    
    //This function was written following the approach of Torun et al. 2012:
    //"Novel GPU Implementation of Jacobi Algorithm for Karhunen-Loeve Transform of Dense Matrices"    
    //P is the number of rows and cols of A
    //Nk is the number of PxP matrices to be diagonalized
    //A is the input 3D matrix (input as a linear array), each sequence of P*P elements represents a PxP matrix expressed in row major format)
    //NOTE: A is not a row major 3d matrix
    //e.g. for P=2,Nk=2: indexing [0][0][0], [0][1][0], [1][0][0], [1][1][0], [0][0][1], [0][1][1], [1][0][1], [1][1][1] 
    //                            --------------First matrix to diag--------  -------- Second matrix to diag ----------   
    
    int MethodFlag = 3; //1- SA, 2-MCA, 3-OSPJ
    //SA - "Symmetric Accss"
    //MCA - "Maximum Coalesced Access"
    //OSPJ - "One Step Parallel Jacobi" - Fastest (1 Kernel call, fewer Global mem access), Most mem efficient (fewer global variables)
    
    if (P>MAX_BLOCK_SIZE){printf("Matrix is too large for current implementation (# rows is g.t. max block size). \n"); exit(0);}
    if (P%2!=0){printf("Matrix input has odd number of rows - not supported. Pad the matrix with zeros and re-run. \n"); exit(0);}
    
    size_t floating_PP = sizeof(floating)*P*P; // size_t is the data type returned by sizeof function
    size_t floating_PPNk = sizeof(floating)*P*P*Nk;       

    floating *device_A, *device_X, *device_eigenvectors_T; //after decomp, the diagonals of device_A will be the eigenvalues,    
    gpuErrchk(cudaMalloc((void **)&device_A, floating_PPNk));
    gpuErrchk(cudaMalloc((void **)&device_X, floating_PPNk));
    gpuErrchk(cudaMalloc((void **)&device_eigenvectors_T, floating_PPNk));
    gpuErrchk(cudaMemcpy(device_A, A, floating_PPNk, cudaMemcpyHostToDevice));    
       
    int *device_IterBlockToElem; //to store mapping of P/2 "blocks" to element at (p,q), computed in the first kernel call    
    gpuErrchk(cudaMalloc((void **)&device_IterBlockToElem, sizeof(int) *(P-1)*P / 2 * 2));     
    kernel_compute_all_chess_params<<<P-1, P/2,0>>>(P, Nk, device_IterBlockToElem);
    
    int *device_fmapping;
    gpuErrchk(cudaMalloc((void **)&device_fmapping, sizeof(int) *(P-1)*P / 2 * 2));   
    kernel_compute_fmapping<<<P-1, P/2, 2*P*sizeof(int)>>>(P, device_IterBlockToElem, device_fmapping);      
    //cudaDeviceSynchronize();
    
     /*
    int *temp;//debugging
    temp = (int *)malloc(sizeof(int)*(P-1)*P/2*2);
    cudaMemcpy(temp,device_IterBlockToElem, sizeof(int)*(P-1)*P/2*2, cudaMemcpyDeviceToHost);
    printf("Chess params: \n"); for (int i = 0; i< (P-1)*P/2*2; i++){printf("%d \n",temp[i]);}
    //printf("Chess params and A value: \n"); for (int i = 0; i< (P-1)*P/2*2; i++){printf("%d_%f\n",temp[i],A[temp[i]]);}
    int *temp2;
    temp2 = (int *)malloc(sizeof(int)*(P-1)*P/2*2);
    cudaMemcpy(temp2,device_fmapping, sizeof(int)*(P-1)*P/2*2, cudaMemcpyDeviceToHost);
    printf("Chess params, fmapping: \n"); for (int i=0; i<(P-1)*P/2*2; i++ ) {printf("%d, %d \n", temp[i], temp2[i]);}    
     */
       
    //Preamble - intialization
    s_initialize_3Didentity(eigenvectors_T, P, Nk); //set eigenvector matrices to identity ** shouldn't this be done on the GPU?
    gpuErrchk(cudaMemcpy(device_eigenvectors_T, eigenvectors_T, floating_PPNk, cudaMemcpyHostToDevice));
    floating *device_sine, *device_cosine, *device_offset_;
    gpuErrchk(cudaMalloc((void **)&device_sine, floating_PPNk));
    gpuErrchk(cudaMalloc((void **)&device_cosine, floating_PPNk));
    cudaMemset(device_sine, 0, floating_PPNk); //initialize array values to zeros
    cudaMemset(device_cosine, 0, floating_PPNk);
    gpuErrchk(cudaMalloc((void **)&device_offset_, sizeof(floating)));
    
    /*
    //Test: using GPU to initialize the identity matrix
    cudaMemset(device_eigenvectors_T, 0, floating_PPNk);
    dim3 blockDim(8,8,16); //ideally, should be multiples of 32
    dim3 gridDim((P + blockDim.x - 1) / blockDim.x, (P + blockDim.y - 1) / blockDim.y, (Nk + blockDim.z -1) / blockDim.z); //total number of threads = gridDim(1)*gridDim(2)*gridDim(3);
    kernel_s_initialize_3Didentity<<<gridDim, blockDim>>>(device_eigenvectors_T, P, Nk);
    //#threads in a block= blockDim.x * blockDim.y * blockDim.z
    //#blocks in grid = gridDim.x * gridDim.y * gridDim.z
    //total # threads = # blocks in grid * # threads in block
    //idea1 - launch as many threads as there are elements in matrix being initialized
    //idea2 - since only diagonals are being set, only launch as many threads as there are diagonal elements?    
        //debugging - check result of GPU identity matrix:
        gpuErrchk(cudaMemcpy(eigenvectors_T, device_eigenvectors_T, floating_PPNk, cudaMemcpyDeviceToHost));
        print_3Dmatrix(eigenvectors_T, P, P, Nk, 1);
    */
       
    //Jacobi segment begins
    int grid_size, block_size=P, iter = 0, counter = 0; //this is the grid_size and block_size that would be used for a single PxP matrix    
    floating offset_ = THRESHOLD + 1;
    if (P%2==0)
        grid_size = P / 2;
    else
        grid_size = P/2 + 1;
    
    bool alg_converged = false;
    bool *device_alg_converged;
    gpuErrchk(cudaMalloc((void **) &device_alg_converged, sizeof(bool)))
    
    while (counter < MAX_SWEEPS && !alg_converged) //sweeps
    //while (counter < MAX_SWEEPS && offset_ > THRESHOLD) //sweeps
    //while (counter < MAX_SWEEPS) //sweeps, neglecting convergence
    {
        iter = 0;
        while (iter < P - 1)
        {
            //printf("iter: %d \n", iter);
            //Compute rotation parameters for all (p,q): q>p
            kernel_compute_params<<<Nk, grid_size>>>(device_A, P, iter, device_sine, device_cosine, device_IterBlockToElem); //calculates sine and cosine params
            //cudaDeviceSynchronize(); //blocks host until all CUDA calls are complete            
            //if each kernel is on the same stream, then don't think DeviceSynchronize is necessary
            
              /* //debugging:
                floating *temp;
                temp = (floating *)malloc(floating_PPNk);
                if (iter == 0 && counter==0){
                    cudaMemcpy(temp,device_sine, floating_PPNk, cudaMemcpyDeviceToHost);
                    printf("sine, iter=0, counter=0 \n");  print_3Dmatrix(temp,P,P,Nk,1);
                }
              */    
             
            if (MethodFlag==1 | MethodFlag==2){  
            
                if (MethodFlag==1){
                //row-update kernel
                kernel_row_update<<<Nk*grid_size, block_size>>>(iter, device_A, device_X, P, device_sine, device_cosine, device_IterBlockToElem);                
                }
            
             /* //debugging - check row update output matrix (X)
            floating *temp; temp = (floating *)malloc(floating_PPNk);
            //if (iter ==0 && counter ==1){
            if (counter==0){
                cudaMemcpy(temp,device_X, floating_PPNk, cudaMemcpyDeviceToHost);
                printf("device_X (original), iter=0, counter=0 \n");  print_3Dmatrix(temp,P,P,Nk,1); printf("\n");
            }
             */
                
                if (MethodFlag==2){             
                    kernel_row_updateMCA<<<Nk*grid_size, block_size, 4*P*sizeof(floating)>>>(iter, device_A, device_X, P, device_sine, device_cosine, device_IterBlockToElem, device_fmapping);
                }
                
             /* //debugging
            floating *temp2; temp2 = (floating *)malloc(floating_PPNk);
            //if (iter ==0 && counter ==1){
            if (counter==0){
                cudaMemcpy(temp2,device_X, floating_PPNk, cudaMemcpyDeviceToHost);
                printf("device_X (MCA), iter=0, counter=0 \n");  print_3Dmatrix(temp2,P,P,Nk,1); printf("\n");
            }
             */
                                   
            
            //col-update & eigen-vector update kernel
            kernel_col_update<<<Nk*grid_size, block_size>>>(iter, device_A, device_X, P, device_eigenvectors_T, device_sine, device_cosine, device_IterBlockToElem);
            
            //printf("iter: %d \n", iter);
            }
            
            
            if (MethodFlag==3){
                kernel_OneStepUpdate<<<Nk*grid_size, block_size, 4*P*sizeof(floating)>>>(iter, device_A, P, device_eigenvectors_T, device_sine, device_cosine, device_IterBlockToElem, device_fmapping);
            }
                        
            //if (iter==2){printf("EXITING EARLY \n"); exit(0);}
            
             /* //debugging:
            if (iter<2)
            {                
               cudaMemcpy(temp,device_A, floating_PPNk, cudaMemcpyDeviceToHost); 
               printf("matrix A: \n");
               print_3Dmatrix(temp,P,P,Nk,1);
            }
             */
             
             
            
            iter++;
        }        
        /*
        cudaMemcpy(A, device_A, floating_PPNk, cudaMemcpyDeviceToHost);         
        offset_ = compute_offset3Dmatrix(A, P, Nk);
        alg_converged = offset_ < THRESHOLD;
        printf("Sweep:%d, offset:%f\n", counter, offset_);
        */ 
        
        kernel_compute_offset3Dmatrix_converge<<<Nk,MAX_BLOCK_SIZE>>>(device_A, P, Nk, device_offset_, device_IterBlockToElem, device_alg_converged);
        cudaMemcpy(&alg_converged, device_alg_converged, sizeof(bool), cudaMemcpyDeviceToHost);
        printf("Sweep:%d \n", counter);
        
        //print_matrix(A,P,P,1); //debugging                
        counter++;            
    }
    cudaMemcpy(A, device_A, floating_PPNk, cudaMemcpyDeviceToHost);
    cudaMemcpy(eigenvectors_T, device_eigenvectors_T, floating_PPNk, cudaMemcpyDeviceToHost);
    
    // /*
    //printf("offset orig %f \n", offset_); 
    //kernel_compute_offset3Dmatrix<<<Nk*P/2,P-1>>>(device_A, P, Nk, device_offset_, device_IterBlockToElem); //want it to output the offset value
    //cudaMemcpy(&offset_, device_offset_, sizeof(floating), cudaMemcpyDeviceToHost);
    //printf("offset new %f \n", offset_);
    
    kernel_compute_offset3Dmatrix_converge<<<Nk,MAX_BLOCK_SIZE>>>(device_A, P, Nk, device_offset_, device_IterBlockToElem, device_alg_converged); 
    cudaMemcpy(&alg_converged, device_alg_converged, sizeof(bool), cudaMemcpyDeviceToHost);   
    //printf("alg converged state= %d \n", alg_converged);    
    // */
}

////////////////////
int main()
{
   //A: sequence of 2D matrices to be decomposed 
   //P: # of rows and cols of the symmetric matrix 'A' 
   //Nk: # of PxP matrices that A is composed of 
       
   high_resolution_clock::time_point t_begin, t_end, t1, t2,t3;    
   duration<floating> time_span, time_span2; 
   
   //Test cases: 
   //floating A[9] = {2, -4, 1, -4, 5, 1, 1, 1, 2}; 
   //int P=3; 
   
   //floating A[16] = {2,-4,1,0,-4,5,1,0,1,1,2,0,0,0,0,0}; 
   //int P=4;
   
   /*
    floating A[16] = {4, -30, 60, -35, -30, 300, -675, 420, 60, -675, 1620, -1050, -35, 420, -1050, 700}; // matrix to be decomposed 
   int P=4; //# of rows of matrix 'A'
   int Nk=1;
   */
   
   //floating A[25]={1, 2, 3, 4, 5, 2, 4, 6, 8, 10, 3, 6, 9, 12, 15, 4, 8, 12, 16, 20, 5, 10, 15, 20, 25}; 
   //int P=5;
   
   //floating A[36]={2875,1762,2671,1498,2221,1294,1762,2677,1978,2113,1462,2329,2671,1978,2659,1510,2005,1498,1498,2113,1510,2551,2086,2563,2221,1462,2005,2086,2677,1870,1294,2329,1498,2563,1870,2767};
   //int P=6; 
   
    /* 
   floating A[64*2]={14764,3928,5720,9388,7596,11096,12888,2220,3928,11692,10412,7768,9048,6572,5292,12888,5720,10412,9644,8024,8792,7340,6572,11096,9388,7768,8024,8620,8364,8792,9048,7596,7596,9048,8792,8364,8620,8024,7768,9388,11096,6572,7340,8792,8024,9644,10412,5720,12888,5292,6572,9048,7768,10412,11692,3928,2220,12888,11096,7596,9388,5720,3928,14764,
       14764,3928,5720,9388,7596,11096,12888,2220,3928,11692,10412,7768,9048,6572,5292,12888,5720,10412,9644,8024,8792,7340,6572,11096,9388,7768,8024,8620,8364,8792,9048,7596,7596,9048,8792,8364,8620,8024,7768,9388,11096,6572,7340,8792,8024,9644,10412,5720,12888,5292,6572,9048,7768,10412,11692,3928,2220,12888,11096,7596,9388,5720,3928,14764}; 
   int P=8;
   int Nk = 2;
    */
   
    /*
   floating A[64*3]={14764,3928,5720,9388,7596,11096,12888,2220,3928,11692,10412,7768,9048,6572,5292,12888,5720,10412,9644,8024,8792,7340,6572,11096,9388,7768,8024,8620,8364,8792,9048,7596,7596,9048,8792,8364,8620,8024,7768,9388,11096,6572,7340,8792,8024,9644,10412,5720,12888,5292,6572,9048,7768,10412,11692,3928,2220,12888,11096,7596,9388,5720,3928,14764,       
   1.920188e+04,2.058504e+04,2.241607e+04,2.293878e+04,1.088616e+04,1.268967e+04,1.543893e+04,1.634972e+04,2.058504e+04,2.836570e+04,2.732219e+04,2.892768e+04,1.331679e+04,1.554238e+04,2.046872e+04,1.986829e+04,2.241607e+04,2.732219e+04,3.021303e+04,2.993872e+04,1.197496e+04,1.546113e+04,1.779099e+04,1.928698e+04,2.293878e+04,2.892768e+04,2.993872e+04,3.312610e+04,1.503402e+04,1.853382e+04,2.061623e+04,2.120199e+04,1.088616e+04,1.331679e+04,1.197496e+04,1.503402e+04,1.557441e+04,9.832584e+03,1.837476e+04,1.463367e+04,1.268967e+04,1.554238e+04,1.546113e+04,1.853382e+04,9.832584e+03,1.741195e+04,1.090660e+04,1.485545e+04,1.543893e+04,2.046872e+04,1.779099e+04,2.061623e+04,1.837476e+04,1.090660e+04,2.455812e+04,1.826495e+04,1.634972e+04,1.986829e+04,1.928698e+04,2.120199e+04,1.463367e+04,1.485545e+04,1.826495e+04,2.055110e+04,
       2.342654e+02,1.011793e+02,1.687537e+02,2.222325e+02,2.418634e+02,2.057266e+02,1.404922e+02,1.455200e+02,1.011793e+02,2.284681e+02,1.618602e+02,1.875495e+02,2.056183e+02,1.537151e+02,1.075004e+02,1.958098e+02,1.687537e+02,1.618602e+02,2.009812e+02,2.060105e+02,2.332961e+02,2.026605e+02,1.534406e+02,1.831566e+02,2.222325e+02,1.875495e+02,2.060105e+02,2.801229e+02,2.819142e+02,2.334663e+02,1.741470e+02,1.958475e+02,2.418634e+02,2.056183e+02,2.332961e+02,2.819142e+02,3.320798e+02,2.763884e+02,1.829447e+02,2.497840e+02,2.057266e+02,1.537151e+02,2.026605e+02,2.334663e+02,2.763884e+02,2.940684e+02,1.647703e+02,2.717750e+02,1.404922e+02,1.075004e+02,1.534406e+02,1.741470e+02,1.829447e+02,1.647703e+02,1.347530e+02,1.278570e+02,1.455200e+02,1.958098e+02,1.831566e+02,1.958475e+02,2.497840e+02,2.717750e+02,1.278570e+02,3.152321e+02}; 
   int P=8;
   int Nk = 3;
    */
   
    /*
   int P=128, Nk=2;
   double A[128*128*2];
   FILE *read_ptr;
   read_ptr = fopen("test2Rs.bin","rb");
   fread(A,sizeof(A),1,read_ptr);
   fclose(read_ptr);   
    */

   // /*
   printf("testing \n");
   int P=128, Nk=16; //1000;
   floating *A;
   float *Atemp;
   A = (floating *)malloc(sizeof(floating)*P*P*Nk);
   Atemp = (float *)malloc(sizeof(float)*P*P*Nk);
   FILE *read_ptr;
   read_ptr = fopen("../../TestData/test4Rs.bin","rb"); //this file contains single precision data, of 1331 matrices each 128x128
   fread(Atemp,sizeof(float),P*P*Nk,read_ptr);  
   fclose(read_ptr); 
   #ifdef USE_DOUBLE_PRECISION //if A is of type double, then convert Atemp to double and store in A      
   for (int i=0; i<P*P*Nk; i++){A[i]=(floating) Atemp[i];}
   #else //just copy Atemp into A
   memcpy(A,Atemp,sizeof(float)*P*P*Nk);   
   #endif
   // */
   
   floating *eigenvectors_T;
   eigenvectors_T = (floating *)malloc(sizeof(floating)*P*P*Nk);
   
   printf("Input matrix:\n");
   //print_3Dmatrix(A, P, P, Nk, 1); //print_matrix(A, P, P, 1);
     
   
   //s_initialize_3Didentity(A, P, Nk);//test identity creation  
   //print_3Dmatrix(A, P, P, Nk, 1);
   //return 0;
   
   t_begin = high_resolution_clock::now();
   
   cudaProfilerStart();
   EigJacobi(P, Nk, A, eigenvectors_T);
   cudaProfilerStop();
   
   t_end = high_resolution_clock::now();
   time_span = duration_cast<duration<floating>>(t_end - t_begin);
   printf("TOTAL TIME (s):%f\n", time_span.count());  
   
   printf("eigenvalues: \n");
   //print_3Dmatrix(A, P, P, Nk, 1);
   printf("eigenvectors: \n");
   //print_3Dmatrix(eigenvectors_T, P, P, Nk, 1);   
   
   // /*
   printf("Just the eigenvalues \n");
   for (int i =0; i<P; i++){
       printf("%f \n", A[i*P + i]);
   }
   // */
   
   cudaDeviceReset();
   return 0;
}


//nvcc -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin" -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" Jacobi3D_CUDA_SP_MCA.cu -o Jacobi3D_CUDA_SP_MCA
