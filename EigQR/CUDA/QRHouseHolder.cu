//At present, I've only coded this to operate on Real symmetric matrices (can be extended to imaginary symmetric matrices later)
//Following the work of Alain Cosnuau, 2014 "Computation on GPU of Eigenvalues and Eigenvectors of a large number of small hermitian matrices"
//Note: this approach isn't really a parallel eig solver (like jacobi method), it's basically the same algorithm as would be implemented on CPUs, it's just that matrix multiplications are done in parallel and each matrix decomposition is assigned to a different block

#include <assert.h>
#include <chrono> //for timing
using namespace std::chrono;
#include<cuda_profiler_api.h> 

#define USE_DOUBLE_PRECISION // to switch between single and double precision, can just comment this out and recompile
#ifdef USE_DOUBLE_PRECISION
typedef double floating;
typedef double2 floating2;
#else
typedef float floating;
typedef float2 floating2;
#endif

#define MAX_ITER 400
#define QRTOL 1e-9 //Value determining convergence of algorithm 
#define MAX_BLOCK_SIZE 1024

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

void s_transpose(floating *M, int m, int n, floating *M_T)
{
    int i, j, index_;
    for (j=0; j<n; j++)
    {
        index_ = j*m;
        for (i=0; i<m; i++)
        {
            M_T[index_+i] = M[i*n+j];
        }
    }
}

void s_multiply(floating *M_1, int m1, int n1, floating *M_2, int m2, int n2, floating *result)
{
    assert(n1 == m2);
    floating sum = 0.0;
    //compute M_2_T:
    floating *M_2_T = (floating *)malloc(sizeof(floating) * n2 * m2);
    s_transpose(M_2, m2, n2, M_2_T);
    int i, j, k, temp1, temp2;
    for (i = 0; i < m1; i++)
    {
        temp1 = i * n1;
        for (j = 0; j < n2; j++)
        {
            sum = 0.0;
            temp2 = j * m2;
            for (k = 0; k < n1; k++)
            {
                sum += M_1[temp1 + k] * M_2_T[temp2 + k];
            }
            result[i * n2 + j] = sum;
        }
    }
    free(M_2_T);
}

floating l2_matrix_diff_norm(floating *E_, floating *E, int M, int N)
{
    floating maxdiff;
    int imax,jmax;
    maxdiff=0.0;
    
    floating sum = 0.0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            sum += (E_[i * M + j] - E[i * M + j]) * (E_[i * M + j] - E[i * M + j]);
            if (maxdiff < fabs((E_[i * M + j] - E[i * M + j]))){
                maxdiff = fabs((E_[i * M + j] - E[i * M + j]));
                imax=i;
                jmax=j;
            }
        }
    }
    printf("max difference is %f \n", maxdiff);
    printf("i,j= %d,%d \n",imax,jmax);
    return sqrt(sum);
}

/*
__device__ void house(x,n,k){
    //computes the householder vector
    //my intention here is to take in a shared variable, modify it and output it as the HH vector
    localId=threadIDx.x;
    if (localID<k){//*** dbl theck this indexing
        x(localID)=0; //i don't think this should be necessary
    }
    if (localID>=k){
        
    }    
}
*/

__device__ int NearestPowerOf2 (int n){
    if (!n) return n; // (0==2^0)
    
    int x = 1;
    while(x<n)
    {
        x<<=1;
    }
    return x;
}

__device__ floating ParallelReduce (floating *stemp, int n){
    //parallel reduce 
    //stemp is a shared variable of 'n' elements
    //n must be a power of 2
    
    floating sum;
    int j = threadIdx.x;
    for (int s=blockDim.x/2; s>0; s>>=1){
        if (j<s) {
            stemp[j] += stemp[j+s];                    
        }            
        __syncthreads();
    }
    sum = stemp[0];
    return sum;
}

__device__ floating ParallelReduceSubset (floating *stemp, int n, int k){
    //parallel reduce, but only for elements k->n-1 of the shared variable (result stored in kth element)
    //stemp is a shared variable of 'n' elements 
    //in this implementation neither 'n' nor 'n-k' need to be powers of 2 
    
    floating sum;
    int nTotalThreads = NearestPowerOf2(n-k); //total number of threads, rounded up to the next power of 2 
    //printf("ntotalthreads %d \n",nTotalThreads);
    int j = threadIdx.x;
    for (int s=nTotalThreads/2; s>0; s>>=1){
        //if (localID==k){printf("s %d \n", s);}
        if (j<s+k) { //s+k is the halfPoint
            int thread2 = j + s;
            if (thread2<blockDim.x) //skipping fictitious threads
                stemp[j] += stemp[thread2]; // pairwise summation
        }            
        __syncthreads();
    }             
    sum = stemp[k];
    return sum;
}

__device__ int sign(floating x) 
{//note if x=0, then the result is 0
    int t = x<0 ? -1 : 0; //reminder to me: this means if (x<0), then t=-1, else, t=0
    return x>0 ? 1 : t;
}

__device__ floating WilkinsonShift (floating a1, floating a2, floating b1){
    //Inputs are the elements comprising the matrix:
    //[a1 b1]
    //[b1 a2]
    //often written as:
    //[a(n-2) b(n-2)]
    //[b(n-2) a(n-1)]
    
    //Output: the shift mu, nearest eigenvalue to d;  
    //mu is that eigenvalue of the tridiagonal matrix T's trailing 2-by-2 principal submatrix closer to t(n,n)
    
    //Wilkinson shift:
    floating mu;
    floating d = (a1 - a2)/2.0;
    if (d==0.)
        mu = a2 - fabs(b1);
    else
        //mu = a2 - b1*b1/(d + d/fabs(d)*sqrt(d*d + b1*b1));
        mu = a2 - (b1*b1)/(d + sign(d)*sqrt(d*d + b1*b1));
    return mu;
}

__global__ void kernel_TriDiagonalize(floating *A, floating *Q, floating *a, floating *b, int n)
{
    //Transforms the nxn hermitian matrix A into hermitian tridiagonal matrix
    //Q is the accumulation of similarity tranformations (the Q input to this function is expected to be the identity matrix)
    //Note, the updated matrix A at each step 'k' is hermitian
        
    //A,Q matrix are stored in global memory (n x n x Nk) //Nk is number of matrices
    //u,w are 1d arrays stored in shared memory (each has size 'n')
    //p - vector of size 'n' stored in shared memory
    //sig, alpha - shared vars
    //stemp - shared
    
    int localID = threadIdx.x;
    int blockID = blockIdx.x; 
    
//     extern __shared__ floating sharedarray[];
//     floating *u = (floating *) sharedarray;
//     floating *w = (floating *) &u[n];
//     floating *p = (floating *) &w[n];
//     floating *stemp = (floating *) &p[n];
//     __shared__ floating u[5];
//     __shared__ floating w[5];
//     __shared__ floating p[5];
//     __shared__ floating stemp[5];
    
    extern __shared__ floating sharedarray[];
    floating *u = (floating *) sharedarray;
    floating *w = (floating *) &u[n];
    floating *stemp = (floating *) &w[n];
    floating p; //Cosnuau 2014 declared this a shared var, but I don't see why
    
    __shared__ floating sig, alpha;
    __shared__ floating v1;    
      
    floating v, sig2;
    
    //store tridiagonal matrix 'T' in the form of its main diagonal (a) and sub and super diagonals (b)
    int k;
    for (k=0; k<n-2; k++){
    //for (k=0; k<32; k++){ //***testing, delete this!
        //if (localID==k){printf("\n k = %d \n",k);}
        
        //technically, each loop, less and less threads are in use (i.e. n-k # threads: localID>=k)
                        
        //STEP 1-form the householder vector
            u[localID] = A[k*n + localID + n*n*blockID]; //read in the vector used to form HH vec (note: technically this should read in the column of A, but since A is symmetric on every k-iter, for coalesced glob read can just read in row;            
            
            if (localID<=k+1){
                stemp[localID]=0.;                
            }
            else{        
                stemp[localID] = u[localID]*u[localID];                
            }
            __syncthreads();
            
            //printf("u[%d] %f \n ", localID, u[localID]);
            //printf("stemp[%d] %f \n ", localID, stemp[localID]);
                        
            //parallel reduce, but only for elements k->n-1 of the shared variable (ans in kth element)
            sig2 = ParallelReduceSubset(stemp, n, k); //note stemp is altered here
            
            //if (localID==k){printf("sig2 %f \n", sig2);}
            
            if (localID==k){                
                u[k]=0.0;
                if (sig2==0.0){
                    sig=0.0;
                    u[k+1] = 1.0;
                    v1 = 1.0;
                }
                else{                    
                    floating x1=u[k+1];
                    if (u[k+1] <= 0.0){
                        v1 = x1 - sqrt(x1*x1 + sig2);
                    }
                    else{
                        v1 = -sig2/(x1 + sqrt(x1*x1 + sig2));
                    }
                    sig=2.0*v1*v1/(sig2 + v1*v1);
                    u[k+1] = v1;
                }
            }
            __syncthreads();            
            u[localID]=u[localID]/v1;            
            __syncthreads(); 
            
            //if (localID==k){printf("House Holder vector: \n");}
            //printf("u[%d]= %f \n",localID, u[localID]);
            
        //STEP 2-calculate v=-sig*A*u                        
            v = 0.0;
            for (int i=k; i<n; i++){
                v += -sig*A[i*n + localID + n*n*blockID]*u[i];
            }
            
            //if (localID==k){printf("V vector: \n");}
            //printf("v[%d]= %f \n",localID, v);
            
            //calculate alpha via sum reduction
            stemp[localID] = -0.5*sig*v*u[localID];
            __syncthreads(); 
            
            //printf("stemp %f \n ", stemp[localID]);
            
            alpha = ParallelReduceSubset(stemp, n, k);
            
            //if (localID==k){printf("after parallel reduction \n");}            
            //printf("stemp %f \n ", stemp[localID]);
            
            w[localID] = v + alpha*u[localID]; 
            __syncthreads(); 
            
            //if (localID==k){printf("W vector: \n");}
            //printf("w[%d]= %f \n",localID, w[localID]);
                       
            
        //STEP 3-update A (             
            for (int i=k; i<n; i++){ //update one row at a time
                A[i*n + localID + n*n*blockID] = A[i*n + localID + n*n*blockID] + w[i]*u[localID] + u[i]*w[localID];                                 
            }           
            
            if (localID==k){
                a[k + n*blockID] = A[k*n + k + n*n*blockID];
                b[k + (n-1)*blockID] = A[k*n + k+1 + n*n*blockID];
            }
                                   
        //STEP 4-update Q - using all threads now
        //p[localID]=0.0; 
        p=0.0;
        
        //printf("p[%d] = %f \n",localID,p[localID]);
        
        for (int i=k+1; i<n; i++){
            //p[localID] = p[localID] + u[i]*Q[i*n + localID + n*n*blockID]; 
            p = p + u[i]*Q[i*n + localID + n*n*blockID];
        }        
        
        //printf("p[%d] = %f \n",localID,p[localID]);
        
        for (int i=k+1; i<n; i++){
            //Q[i*n+localID + n*n*blockID] = Q[i*n+localID + n*n*blockID] - sig*u[i]*p[localID];
            Q[i*n+localID + n*n*blockID] = Q[i*n+localID + n*n*blockID] - sig*u[i]*p; 
        }
        __syncthreads(); //this syncthreads is very necessary because the u[i] could be changed by a thread on the next iter 
        //note: for speed, could get around this syncthreads, if a new shared variable other than u was used to store the globally read in data from A at the begginning of each iter
        
        //printf("Q[%d]=%f \n",(k)*n + localID, Q[(k)*n + localID]);
    } //k loop
    
    //printf("Q[%d]=%f \n",(1)*n + localID, Q[(1)*n + localID]);    
    
    if (localID==0){//assign the elements of the remaining 2x2 matrix of A        
        a[n-2 + n*blockID]=A[(n-2)*n + n-2 + n*n*blockID];
        a[n-1 + n*blockID]=A[(n-1)*n + n-1 + n*n*blockID];
        b[n-2 + (n-1)*blockID]=A[(n-2)*n + n-1 + n*n*blockID];
    }
    
    //__syncthreads(); printf("b[%d]= %f \n",localID, b[localID]);
}

////////////////////////////////////////
__global__ void kernel_DiagonalizeHH(floating *a, floating *b, floating *Q, int n)
{
    //This function diagonalizes the real, tri-diagonal matrix (n x n) given by vars a and b
    //Input:
    //a - the main diagonal (n elements)
    //b - the first sub (and super) diagonal (n-1 elements)
    //Output:
    //a - eigenvalues
    //Q - eigenvectors
    
    //Approach:
    //single thread of each block used to diagonalize 
    //all threads of each block used to update e-vecs (Q)
    
    //This function uses the HouseHolder method
    int localID = threadIdx.x;
    int blockID = blockIdx.x;
    
    __shared__ floating sig, u1;
    floating S;
    
    //the following declarations are really only needed in thread 0... but not sure how to only declare it there or if it's even worth it???? ***
    //if (localID==0){
    floating ck, x1, x2, nu, alpha, mu;
    floating u[4], v[4], w[4];
    //}
        
    int m=n; //initial problem size prior to any deflation
    int iter=0; //'global' iter    
    while (m>1 && iter<MAX_ITER){         
        //if (localID==0) {printf("m %d \n", m);} __syncthreads(); //WTF, with this print statement in, a syncthreads is necessary??
        
        iter++;
        ck=0.; //every iter, the matrix is assumed to be tridiagonal
        mu = WilkinsonShift(a[m-2 + n*blockID], a[m-1 + n*blockID], b[m-2 + (n-1)*blockID]);
        //printf("b %f \n ",b[localID]);
        //printf("mu %f \n",mu);
        
        a[localID + n*blockID] = a[localID + n*blockID] - mu;
        __syncthreads();
        
        for (int k=0; k<m-1; k++){
            if (localID==0){//QR-Householder single threaded version:                
                x1=a[k + n*blockID];
                x2=b[k + (n-1)*blockID];
                
                //nu = x1/fabs(x1) * sqrt(x1*x1 + x2*x2);
                nu = sign(x1) * sqrt(x1*x1 + x2*x2);  // accuracy of this sqrt can really matter               
                sig = 1.0 + x1/nu; //(nu + x1)/nu; 
                u1 = x2/(nu+x1); 
                
                if (k==0){ //couldn't write this more concisely???:   
                    u[0] = 1.; u[1]=u1; u[2]=0.; u[3]=0.;
                    v[0] = -sig*(a[k + n*blockID] + b[k + (n-1)*blockID]*u1);
                    v[1] = -sig*(b[k + (n-1)*blockID] + a[k+1 + n*blockID]*u1);
                    v[2] = -sig*(b[k+1 + (n-1)*blockID]*u1);
                    v[3] = 0.;
                    
                    //floating u[] = {1.f, u1, 0, 0};
                    //floating v[] = {-sig*(a[k] + b[k]*u1),     -sig*(b[k] + a[k+1]*u1),   -sig*(b[k+1]*u1),  0};
                }
                else if (k<m-2){
                    u[0]=0; u[1]=1.; u[2]=u1; u[3]=0.;
                    v[0]=-sig*(b[k-1 + (n-1)*blockID] + ck*u1);
                    v[1]=-sig*(a[k + n*blockID] + b[k + (n-1)*blockID]*u1);
                    v[2]=-sig*(b[k + (n-1)*blockID] + a[k+1 + n*blockID]*u1);
                    v[3]=-sig*(b[k+1 + (n-1)*blockID]*u1);
                    
                    //floating u[] = {0, 1.f, u1, 0};
                    //floating v[] = {-sig*(b[k-1] + ck*u1),     -sig*(a[k] + b[k]*u1),     -sig*(b[k] + a[k+1]*u1),   -sig*(b[k+1]*u1)};
                }
                else if (k==m-2){
                    u[0]=0.; u[1]=0.; u[2]=1.; u[3]=u1;
                    v[0]=0.;
                    v[1]=-sig*(b[k-1 + (n-1)*blockID] + ck*u1);
                    v[2]=-sig*(a[k + n*blockID] + b[k + (n-1)*blockID]*u1);
                    v[3]=-sig*(b[k + (n-1)*blockID] + a[k+1 + n*blockID]*u1);
                    
                    //floating u[] = {0, 0, 1.f, u1};
                    //floating v[] = {0,  -sig*(b[k-1] + ck*u1),     -sig*(a[k] + b[k]*u1),     -sig*(b[k] + a[k+1]*u1)};
                }
                
                alpha = -(0.5)*sig*(v[0]*u[0] + v[1]*u[1] + v[2]*u[2] + v[3]*u[3]);
                w[0] = v[0] + alpha*u[0];
                w[1] = v[1] + alpha*u[1];
                w[2] = v[2] + alpha*u[2];
                w[3] = v[3] + alpha*u[3];
                
                //leaving these in verbose form for ease of translation when the input is complex
                if (k==0){
                    a[k + n*blockID] = a[k + n*blockID] + w[0]*u[0] + u[0]*w[0]; 
                    a[k+1 + n*blockID] = a[k+1 + n*blockID] + w[1]*u[1] + u[1]*w[1];
                    b[k + (n-1)*blockID] = b[k + (n-1)*blockID] + w[1]*u[0] + u[1]*w[0];
                    b[k+1 + (n-1)*blockID] = b[k+1 + (n-1)*blockID] + w[1]*u[2] + u[1]*w[2];
                    ck = w[0]*u[2] + u[0]*w[2];
                }
                else if (k<m-2){
                    a[k + n*blockID] = a[k + n*blockID] + w[1]*u[1] + u[1]*w[1];
                    a[k+1 + n*blockID] = a[k+1 + n*blockID] + w[2]*u[2] + u[2]*w[2];
                    b[k + (n-1)*blockID] = b[k + (n-1)*blockID] + w[2]*u[1] + u[2]*w[1];
                    b[k+1 + (n-1)*blockID] = b[k+1 + (n-1)*blockID] + w[3]*u[2] + u[3]*w[2];
                    b[k-1 + (n-1)*blockID] = b[k-1 + (n-1)*blockID] + w[1]*u[0] + u[1]*w[0];
                    ck = w[3]*u[1] + u[3]*w[1];
                }
                else if (k==m-2){
                    a[k + n*blockID] = a[k + n*blockID] + w[2]*u[2] + u[2]*w[2];
                    a[k+1 + n*blockID] = a[k+1 + n*blockID] + w[3]*u[3] + u[3]*w[3];
                    b[k + (n-1)*blockID] = b[k + (n-1)*blockID] + w[3]*u[2] + u[3]*w[2];
                    b[k-1 + (n-1)*blockID] = b[k-1 + (n-1)*blockID] + w[2]*u[1] + u[2]*w[1]; 
                }
                
                /*
                printf("alpha %f \n",alpha);
                for (int i=0; i<4; i++){
                    printf("u(i) %f \n",u[i]);
                }
                */                
            }
            __syncthreads();
            
            //Update Q
            //technically, this updates Qh (the conj transpose of Q, not Q), and is more efficient:
            S = sig*( Q[n*k + localID + n*n*blockID] + u1 * Q[n*(k+1) + localID + n*n*blockID] );
            Q[n*k + localID + n*n*blockID] = Q[n*k + localID + n*n*blockID] - S;
            Q[n*(k+1) + localID + n*n*blockID] = Q[n*(k+1) + localID + n*n*blockID] - u1*S;
            
            //this updates Q:
            //S = sig*( Q[n*localID + k + n*n*blockID] + u1 * Q[n*localID + k+1 + n*n*blockID] );
            //Q[n*localID + k + n*n*blockID] = Q[n*localID + k + n*n*blockID] - S;
            //Q[n*localID + k+1 + n*n*blockID] = Q[n*localID + k+1 + n*n*blockID] - u1*S;
            
            //if (n==m && k==0){printf("Q[%d]=%f \n",(1)*n + localID, Q[(1)*n + localID]);}
            
            //__syncthreads(); //necessary?
        } //k loop
        a[localID + n*blockID] = a[localID + n*blockID] + mu;
        __syncthreads(); 
        
        if (fabs(b[m-1-1 + (n-1)*blockID]) <= QRTOL*fabs(a[m-1 + n*blockID])){
            m = m - 1;
        }        
    } // m while condn
    //if (localID==0){printf("Iterations for Diagonalization: %d \n",iter);}  
    
    //__syncthreads(); printf("Q[%d]=%f \n",(127)*n + localID, Q[(127)*n + localID]);
}

//////////////////////////////
__device__ floating2 myGivens(floating a, floating b){    
    floating2 cs_params; //cs_params.x : stores cos(theta), cs_params.y : stores sin(theta)
    floating tau;
    
    if (b==0) {
        cs_params.x=1.0;
        cs_params.y=0.0;
    }
    else{
        if (fabs(b) > fabs(a)){
            tau = -a/b; 
            cs_params.y = 1.0/sqrt(1+tau*tau); 
            cs_params.x = cs_params.y*tau;
        }
        else{
            tau = -b/a; 
            cs_params.x = 1.0/sqrt(1+tau*tau);
            cs_params.y = cs_params.x*tau;
        }
    }
    return cs_params;
}

__device__ floating2 SymSchur(floating app, floating apq, floating aqp, floating aqq){
    //determine the c,s parameters that will diagonalize the input matrix [app apq; aqp app]
    //i.e. [bpp bpq; bqp bqq] = [c s; -s c]'*[app apq; aqp aqq]*[c s; -s c]; % where the result is diagonal
    floating2 cs_params; //cs_params.x : stores cos(theta), cs_params.y : stores sin(theta)
    floating y,d,r,elem;
    
    elem = apq;
    y = (aqq - app)*0.5;
    d = fabs(y) + sqrt(elem*elem + y*y);
    r = sqrt(elem*elem + d*d);
    if (r<0.0){
        cs_params.x=1.0;
        cs_params.y=0.0;
    }
    else {
        if (y!=0.0){
            cs_params.x = d/r;
            cs_params.y = sign(y) * elem/r;
        }
        else {
            cs_params.x = 1.0/sqrt(2.0);
            cs_params.y = 1.0/sqrt(2.0);
        }
    }    
    return cs_params;
}

__global__ void kernel_DiagonalizeGivens(floating *a, floating *b, floating *Q, int n)
{
    //This function diagonalizes the real, tri-diagonal matrix (n x n) given by vars a and b
    //Input:
    //a - the main diagonal (n elements)
    //b - the first sub (and super) diagonal (n-1 elements)
    //Output:
    //a - eigenvalues
    //Q - eigenvectors
    
    //Approach:
    //single thread of each block used to diagonalize 
    //all threads of each block used to update e-vecs (Q)
    
    //This function uses the givens method
    int localID = threadIdx.x;
    int blockID = blockIdx.x;
    
    __shared__ floating c,s;
    floating x, y, w_giv, d, z, Qtemp1, Qtemp2, mu;
    floating2 cs_params;
           
    //__shared__ floating a_sh[128], b_sh[127]; //static for testing specific case
    extern __shared__ floating sharedarray[];
    floating *a_sh = (floating *) sharedarray;
    floating *b_sh = (floating *) &a_sh[n];
    
    //load from global to shared (n*blockID is the start index of each matrix)
    a_sh[localID] = a[localID + n*blockID]; 
    b_sh[localID] = b[localID + (n-1)*blockID];
    __syncthreads();
    
    int m=n; //initial problem size prior to any deflation
    int iter=0; //'global' iter    
    while (m>1 && iter<MAX_ITER){
        iter++;        
        mu = WilkinsonShift(a_sh[m-2], a_sh[m-1], b_sh[m-2]);
        x = a_sh[0] - mu;
        y = b_sh[0];
        
        for (int k=0; k<m-1; k++){
            if (localID==0){//givens single threaded version:                
                if (m>1){ 
                    cs_params=myGivens(x,y);
                }
                else {
                    cs_params=SymSchur(a_sh[0], b_sh[0], b_sh[0], a_sh[1]);
                }
                c=cs_params.x;
                s=cs_params.y;
                w_giv = c*x - s*y;
                d = a_sh[k] - a_sh[k+1];
                z = (2.0*c*b_sh[k] + d*s)*s;
                a_sh[k] = a_sh[k] - z;
                a_sh[k+1] = a_sh[k+1] + z;
                b_sh[k] = d*c*s + (c*c - s*s)*b_sh[k];
                x = b_sh[k];
                if (k>0){
                    b_sh[k-1] = w_giv;
                }
                if (k< m-2){
                    y = -s*b_sh[k+1];
                    b_sh[k+1] = c*b_sh[k+1];
                }                               
            }
            __syncthreads();
            
            //Update Q
            //technically, this updates Qh (the conj transpose of Q, not Q), and is more efficient:
           Qtemp1 = Q[k*n + localID + n*n*blockID];
           Qtemp2 = Q[(k+1)*n + localID + n*n*blockID];
           Q[k*n + localID + n*n*blockID] = Qtemp1*c - Qtemp2*s;
           Q[(k+1)*n + localID + n*n*blockID] = Qtemp1*s + Qtemp2*c;
            
            //this updates Q (*not tested*):
//             Qtemp1 = Q[n*localID + k + n*n*blockID];
//            Qtemp2 = Q[n*localID + k+1 + n*n*blockID];
//            Q[n*localID + k + n*n*blockID] = Qtemp1*c - Qtemp2*s;
//            Q[n*localID + k+1 + n*n*blockID] = Qtemp1*s + Qtemp2*c;
            
           //__syncthreads(); //might be necessary?
        } //k loop        
        //__syncthreads();  //don't think this is necessary
        
        if (fabs(b_sh[m-1-1]) <= QRTOL*fabs(a_sh[m-1])){
            m = m - 1;
        }        
    } // m while condn 
    a[localID + n*blockID] = a_sh[localID]; //write back to global     

    //if (localID==0){printf("Iterations for Diagonalization: %d \n",iter);}  
    
    //__syncthreads(); printf("Q[%d]=%f \n",(127)*n + localID, Q[(127)*n + localID]);
}

//////////////////
__global__ void kernel_s_initialize_3Didentity(floating *I, int size, int depth){
    int xID = blockDim.x*blockIdx.x + threadIdx.x;
    int yID = blockDim.y*blockIdx.y + threadIdx.y;
    int zID = blockDim.z*blockIdx.z + threadIdx.z;
    
    if (xID < size && yID < size && zID < depth){
        if (xID == yID)
            I[xID*size + yID + size*size*zID] = 1.0; 
    }
}

void EigQR(int n, int Nk, floating *A, floating *eigenvectors_T, floating *a)
{   
    //n is the number of rows and cols of each matrix to be diagonalized
    //Nk is the number of nxn matrices to be diagonalized
    //A is the input 3D matrix (input as a linear array), each sequence of n*n elements represents a nxn matrix (expressed in row major format)
    //NOTE: A is not a row major 3d matrix
    //e.g. for n=2,Nk=2: indexing [0][0][0], [0][1][0], [1][0][0], [1][1][0], [0][0][1], [0][1][1], [1][0][1], [1][1][1] 
    //                            --------------First matrix to diag--------  -------- Second matrix to diag ----------   

    if (n>MAX_BLOCK_SIZE){printf("Matrix is too large for current implementation (# rows is g.t. max block size). \n"); exit(0);}
    
    size_t floating_nn = sizeof(floating)*n*n; // size_t is the data type returned by sizeof function
    size_t floating_nnNk = sizeof(floating)*n*n*Nk; 
    size_t floating_nNk = sizeof(floating)*n*Nk;
    
    floating *device_A, *device_eigenvectors_T, *device_a, *device_b;
    gpuErrchk(cudaMalloc((void **)&device_A, floating_nnNk));
    gpuErrchk(cudaMalloc((void **)&device_eigenvectors_T, floating_nnNk));
    gpuErrchk(cudaMalloc((void **)&device_a, floating_nNk));
    gpuErrchk(cudaMalloc((void **)&device_b, sizeof(floating)*(n-1)*Nk));
    gpuErrchk(cudaMemcpy(device_A, A, floating_nnNk, cudaMemcpyHostToDevice));
    
//     //Preamble - initialization of eigenvectors to identity matrix: (on GPU)
//     cudaMemset(device_eigenvectors_T, 0, floating_nnNk);
//     dim3 blockDim(8,8,16); //ideally, should be multiples of 32
//     dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y, (Nk + blockDim.z -1) / blockDim.z); //total number of threads = gridDim(1)*gridDim(2)*gridDim(3);
//     kernel_s_initialize_3Didentity<<<gridDim, blockDim>>>(device_eigenvectors_T, n, Nk);
//     //#threads in a block= blockDim.x * blockDim.y * blockDim.z
//     //#blocks in grid = gridDim.x * gridDim.y * gridDim.z
//     //total # threads = # blocks in grid * # threads in block
//     //idea1 - launch as many threads as there are elements in matrix being initialized
//     //idea2 - since only diagonals are being set, only launch as many threads as there are diagonal elements?    
//     //debugging - check result of GPU identity matrix:
//     gpuErrchk(cudaMemcpy(eigenvectors_T, device_eigenvectors_T, floating_nnNk, cudaMemcpyDeviceToHost));
//     //print_3Dmatrix(eigenvectors_T, n, n, Nk, 1);
    
    //Preamble - intialization of eigenvectors to identity matrix: (on CPU)
    s_initialize_3Didentity(eigenvectors_T, n, Nk); 
    gpuErrchk(cudaMemcpy(device_eigenvectors_T, eigenvectors_T, floating_nnNk, cudaMemcpyHostToDevice));
        
    
    int grid_size = Nk, block_size = n; // launch as many blocks as matrices to be diagonalized, each block with a number of threads equal to the size of the matrices to be diagonalized
    kernel_TriDiagonalize<<<grid_size, block_size, 3*n*sizeof(floating)>>>(device_A, device_eigenvectors_T, device_a, device_b, n); 
    //kernel_DiagonalizeHH<<<grid_size, block_size>>>(device_a, device_b, device_eigenvectors_T, n);
    kernel_DiagonalizeGivens<<<grid_size, block_size, (n + n-1)*sizeof(floating)>>>(device_a, device_b, device_eigenvectors_T, n);
    
    gpuErrchk(cudaMemcpy(A, device_A, floating_nnNk, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(a, device_a, floating_nNk, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(eigenvectors_T, device_eigenvectors_T, floating_nnNk, cudaMemcpyDeviceToHost));
        
}

////////////////////////////////////
int main()
{
    //A: sequence of 2D matrices to be decomposed
    //n: # of rows and cols of the symmetric matrix 'A'
    //Nk: # of nxn matrices that A is composed of
    
    high_resolution_clock::time_point t_begin, t_end, t1, t2,t3;    
    duration<floating> time_span, time_span2; 
   
   //Setup of matrices to find evalues and evecs of

     /*
     floating A[16] ={4, 1, -2, 2, 1, 2, 0, 1, -2, 0, 3, -2, 2, 1, -2, -1};
     int n = 4; 
     int Nk = 1;
     */

/*    
    floating A[25] = {2875.000000 , 1762.000000 , 2671.000000 , 1498.000000 , 2221.000000 , 1762.000000 , 2677.000000 , 1978.000000 , 2113.000000 , 1462.000000 , 2671.000000 , 1978.000000 , 2659.000000 , 1510.000000 , 2005.000000 , 1498.000000 , 2113.000000 , 1510.000000 , 2551.000000 , 2086.000000 , 2221.000000 , 1462.000000 , 2005.000000 , 2086.000000 , 2677.000000}; 
    int n = 5;
    int Nk =1;
 */ 

 /*
    floating A[12*12] = {115490.000000 , 20164.000000 , 29668.000000 , 86978.000000 , 77474.000000 , 58180.000000 , 67684.000000 , 48962.000000 , 39458.000000 , 96196.000000 , 105700.000000 , 10946.000000 , 20164.000000 , 98210.000000 , 90434.000000 , 43492.000000 , 51268.000000 , 67106.000000 , 59330.000000 , 74596.000000 , 82372.000000 , 36002.000000 , 28226.000000 , 105700.000000 , 29668.000000 , 90434.000000 , 84386.000000 , 47812.000000 , 53860.000000 , 66242.000000 , 60194.000000 , 72004.000000 , 78052.000000 , 42050.000000 , 36002.000000 , 96196.000000 , 86978.000000 , 43492.000000 , 47812.000000 , 74018.000000 , 69698.000000 , 60772.000000 , 65092.000000 , 56738.000000 , 52418.000000 , 78052.000000 , 82372.000000 , 39458.000000 , 77474.000000 , 51268.000000 , 53860.000000 , 69698.000000 , 67106.000000 , 61636.000000 , 64228.000000 , 59330.000000 , 56738.000000 , 72004.000000 , 74596.000000 , 48962.000000 , 58180.000000 , 67106.000000 , 66242.000000 , 60772.000000 , 61636.000000 , 63650.000000 , 62786.000000 , 64228.000000 , 65092.000000 , 60194.000000 , 59330.000000 , 67684.000000 , 67684.000000 , 59330.000000 , 60194.000000 , 65092.000000 , 64228.000000 , 62786.000000 , 63650.000000 , 61636.000000 , 60772.000000 , 66242.000000 , 67106.000000 , 58180.000000 , 48962.000000 , 74596.000000 , 72004.000000 , 56738.000000 , 59330.000000 , 64228.000000 , 61636.000000 , 67106.000000 , 69698.000000 , 53860.000000 , 51268.000000 , 77474.000000 , 39458.000000 , 82372.000000 , 78052.000000 , 52418.000000 , 56738.000000 , 65092.000000 , 60772.000000 , 69698.000000 , 74018.000000 , 47812.000000 , 43492.000000 , 86978.000000 , 96196.000000 , 36002.000000 , 42050.000000 , 78052.000000 , 72004.000000 , 60194.000000 , 66242.000000 , 53860.000000 , 47812.000000 , 84386.000000 , 90434.000000 , 29668.000000 , 105700.000000 , 28226.000000 , 36002.000000 , 82372.000000 , 74596.000000 , 59330.000000 , 67106.000000 , 51268.000000 , 43492.000000 , 90434.000000 , 98210.000000 , 20164.000000 , 10946.000000 , 105700.000000 , 96196.000000 , 39458.000000 , 48962.000000 , 67684.000000 , 58180.000000 , 77474.000000 , 86978.000000 , 29668.000000 , 20164.000000 , 115490.000000};
    int n=12;
    int Nk = 1;
  */ 
    
   // /*
   printf("testing \n");
   int n=128, Nk=16; //256; //1000;
   floating *A;
   float *Atemp;
   A = (floating *)malloc(sizeof(floating)*n*n*Nk);
   Atemp = (float *)malloc(sizeof(float)*n*n*Nk);
   FILE *read_ptr;
   read_ptr = fopen("../../TestData/test4Rs.bin","rb"); //this file contains single precision data, of 1331 matrices each 128x128
   fread(Atemp,sizeof(float),n*n*Nk,read_ptr);   
   fclose(read_ptr);
   #ifdef USE_DOUBLE_PRECISION //if A is of type double, then convert Atemp to double and store in A      
   for (int i=0; i<n*n*Nk; i++){A[i]=(floating) Atemp[i];}
   #else //just copy Atemp into A
   memcpy(A,Atemp,sizeof(float)*n*n*Nk);   
   #endif
   // */
   
   floating *eigenvectors_T;
   eigenvectors_T = (floating *)malloc(sizeof(floating)*n*n*Nk);
   floating *a;
   a = (floating *)malloc(sizeof(floating)*n*Nk);
   
   if (n<32){
    printf("Input matrix:\n");
    print_3Dmatrix(A, n, n, Nk, 1); //print_matrix(A, P, P, 1);
   }   
      
   //s_initialize_3Didentity(A, P, Nk);//test identity creation  
   //print_3Dmatrix(A, P, P, Nk, 1);
   //return 0;
   
   
   //Hold onto a copy of the input:
   floating *Ain;
   Ain = (floating *)malloc(sizeof(floating)*n*n*Nk);
   memcpy(Ain,A,sizeof(floating)*n*n*Nk);
   
   cudaProfilerStart();
   t_begin = high_resolution_clock::now();
   
   EigQR(n, Nk, A, eigenvectors_T, a); //A becomes tri-diagonalized    
   cudaDeviceSynchronize();   
   
   t_end = high_resolution_clock::now();
   cudaProfilerStop();
   
   time_span = duration_cast<duration<floating>>(t_end - t_begin);
   printf("TOTAL TIME (s):%f\n", time_span.count());  
   
   //printf("Tri-diagonalization result: \n");
   //print_3Dmatrix(A, n, n, Nk, 1);
   
   
//    printf("Eigenvalues \n");
//    for (int i =0; i<n*Nk; i++){
//        printf("%f \n", a[i]);
//    }
   
   
   if (n<32){
    printf("eigenvectors_T: \n");
    print_3Dmatrix(eigenvectors_T, n, n, Nk, 1);   
   }
   
   // /* //something in here is bad:
   printf("Calculating accuracy of method \n"); //Calculate the accuracy of the method:
   //A*V = V*D; //evaluate how close this is to being true (V is evecs, D is evals)
   int iMatrix = 100; //which matrix to calculate accuracy of decomp
   if (iMatrix > Nk-1) { iMatrix = Nk - 1; }
   floating *eigenvectors_test, *eigenvectors_T_test, *eigenvalues_nn_test, *Ain_test;
   eigenvectors_test = (floating *)malloc(sizeof(floating)*n*n); 
   eigenvectors_T_test = (floating *)malloc(sizeof(floating)*n*n); 
   eigenvalues_nn_test = (floating *)malloc(sizeof(floating)*n*n);
   Ain_test = (floating *)malloc(sizeof(floating)*n*n);
    //form matrix of eigenvalues:
   memset(eigenvalues_nn_test, 0.0, sizeof(floating)*n*n);
   for (int i=0; i<n; i++){
       eigenvalues_nn_test[i*n + i] = a[i + n*iMatrix];
       for (int j=0; j<n; j++){
           eigenvectors_T_test[i*n + j] = eigenvectors_T[i*n + j + n*n*iMatrix];
           Ain_test[i*n + j] = Ain[i*n + j + n*n*iMatrix];
       }
   }   
   floating *AV, *VD;
   AV = (floating *)malloc(sizeof(floating)*n*n);
   VD = (floating *)malloc(sizeof(floating)*n*n);
   s_transpose(eigenvectors_T_test, n, n, eigenvectors_test); //take the transpose of evec matrix
   s_multiply(Ain_test, n, n, eigenvectors_test, n, n, AV); //AV 
   s_multiply(eigenvectors_test, n, n, eigenvalues_nn_test, n, n, VD); //VD
   floating accuracy = l2_matrix_diff_norm(AV, VD, n, n);   
   printf("Accuracy of method for matrix # %d = %f \n", iMatrix, accuracy);
   
   /*
   printf("AV \n");
   for (int i =0; i<n; i++)
       printf("%f \n",AV[i*n + 32]);
      
   printf("eigenvectors \n");
   for (int i =0; i<n; i++)
       printf("%f \n",eigenvectors[i*n + 32]);
   */
   
//    if (n<32){
//    printf("eigenvectors: \n");
//    print_3Dmatrix(eigenvectors,n,n,1,1);
//    
//    printf("AV result \n");
//    print_3Dmatrix(AV, n, n, 1, 1);
// 
//    printf("VD result \n");
//    print_3Dmatrix(VD, n, n, 1, 1);
//    }
   
   
   
   cudaDeviceReset();   
    return 0;        
}

//ex: compile with:
//nvcc -lineinfo -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin" -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" QRHouseHolder.cu -o QRHouseHolder -maxrregcount=32

//commandline cuda_profiler_api
//nvprof

//commandline visual profiler:
//nvvp
