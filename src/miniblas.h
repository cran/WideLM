
// Adapted from SDK's 'reduce3' algorithm.
//
// 'blockDim.x' need not be a power of two, although it cannot be
// odd.
//
inline __device__ void OddReduce(unsigned int tid, double *sdata) {
  double tidSum = sdata[tid];
  int lastEnd = blockDim.x;
  for(unsigned int end = blockDim.x >> 1; end > 0; end >>= 1) {
    if (tid == 0) { // Thread zero collects any trailing odd indices.
      double oddEnd = ((lastEnd & 1) == 0) ? 0.0 : sdata[lastEnd-1];
      sdata[0] = tidSum = tidSum + sdata[end] + oddEnd;
    }
    else if (tid < end) {
      sdata[tid] = tidSum = tidSum + sdata[tid + end];
    }
    lastEnd = end;
    __syncthreads();
  }
}

// Returns the sum of the elements of 'dx'.
// 'sdata' is the shared-memory workspace for the reduction.
//
inline __device__ double dcusum(double *dx, double *sdata) {
    unsigned int tid = threadIdx.x;
    sdata[tid] = dx[tid];
    __syncthreads();

    OddReduce(tid, sdata);

    return sdata[0];
}

// Returns the dot product of 'dx' and 'dy'.  Both vectors are accessed
// by unit stride.
//
inline __device__ double dcudot11(double *dx, double *dy, double *sdata) {
  unsigned int tid = threadIdx.x;
  sdata[tid] = dx[tid] * dy[tid];
  __syncthreads();

  OddReduce(tid, sdata);

  return sdata[0];
}

// Returns square of the norm of 'dx'.
//
inline __device__  double dcunrm2(double *dx, double *sdata) {
  unsigned int tid = threadIdx.x;
  sdata[tid] = dx[tid] * dx[tid];
  __syncthreads();

  OddReduce(tid, sdata);

  return sdata[0];
}

//        Form  x :=  A^-1 * x.
//
inline __device__ void dtrsvUNN1(int n, double *a, int lx, double *x) {
  int i, j;
  double temp;
  int colOffs = (n-1)*lx;

  for (j = n-1; j >= 0; j--) {
    temp = x[j]/a[j + colOffs];
    x[j] = temp;
    for (i = 0; i < j; i++) {
      x[i] -= temp * a[i + colOffs];
    }
    colOffs -= lx;
  }
}

inline __device__ void dtrsvUTN1(int n, double *a, int lx, double *x) {
  //        Form  x := (a^T)^-1 * x.

  int i, j;
  double temp;
  int colOffs = 0;
  for (j = 0; j < n; j++) {
    temp = x[j];
    for (i = 0; i < j; i++) {
      temp -= a[i + colOffs]*x[i];
    }
    x[j] = temp/a[j + colOffs];
    colOffs += lx;
  }
}

