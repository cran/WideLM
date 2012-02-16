
//#include<iostream>
using namespace std;
#include <math.h>
#include <cublas.h>
#include "miniblas.h"

const double v1 = sqrt(0.5);

// Holds main constant values.
//
__device__ __constant__ double dv1;
__device__ __constant__ double dv2;
__device__ __constant__ double dRecipDF;
__device__ __constant__ double dw0;
__device__ __constant__ int dnrow;

__global__ void LSFit2Ker(double*, double*, int*, int*, int*, double*, double*, int, int);

// Maximum grid size available:  arch 1.x, 2.x.
//
const uint maxGrid = (1 << 16) - 1;

// 'M' and 'y' assumed to have the same number of rows.
// Adhocracy:  This covers the one or two predictor case only, both additive
// and saturated.
//
__host__ void WideLSF2(double *M, int ncolM, int nrowM, double *y, int ncolY, int *yIdx, int pLength, double *coefs, double *tScores, int *p1, int* p2 = 0, int sat = 0) {
  double *dDesign, *dCoefs, *dTscores, *dY;
  int *dp1, *dp2, *dyIdx;

  int rDim = p2 == 0 ? 2 : (sat ? 4 : 3);
  double recipDF = 1 / double(nrowM - rDim);
  double v2 = 1.0 / sqrt(2.0 * nrowM);
  double w0 = v2 * nrowM;

  // Aligns leading dimension of 'dX' to next warp-size multiple.
  //
  int lx = (nrowM + 31) & (((unsigned) -1) << 5);

  cublasInit();
  // Device constant initializations.
  //
  cudaMemcpyToSymbol(dv1, &v1, sizeof(double), 0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dv2, &v2, sizeof(double), 0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dRecipDF, &recipDF, sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dw0, &w0, sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dnrow, &nrowM, sizeof(int), 0, cudaMemcpyHostToDevice);

  // Overkill, below:  just need to zero the bottom 'lx - nrowM' rows.
  //
  cublasAlloc(lx * ncolM, sizeof(double), (void**) &dDesign);
  cudaMemset2D(dDesign, ncolM * sizeof(double), 0, ncolM * sizeof(double), lx);
  cublasSetMatrix(nrowM, ncolM, sizeof(double), M, nrowM, dDesign, lx);

  cublasAlloc(lx * ncolY, sizeof(double), (void**) &dY);
  cudaMemset2D(dY, ncolY * sizeof(double), 0, ncolY * sizeof(double), lx);
  cublasSetMatrix(nrowM, ncolY, sizeof(double), y, nrowM, dY, lx);

  // All three vectors have the length 'pLength'.
  //
  int vecSize = pLength * sizeof(int);
  cudaMalloc((void**) &dyIdx, vecSize);
  cudaMemcpy(dyIdx, yIdx, vecSize, cudaMemcpyHostToDevice);
  cudaMalloc((void**) &dp1, vecSize);
  cudaMemcpy(dp1, p1, vecSize, cudaMemcpyHostToDevice);
  cudaMalloc((void**) &dp2, vecSize);
  cudaMemcpy(dp2, p2, vecSize, cudaMemcpyHostToDevice);

  int blockSize = rDim * maxGrid * sizeof(double);
  cudaMalloc((void**) &dTscores, blockSize);
  cudaMalloc((void**) &dCoefs, blockSize);

  int curIdx = 0;
  // Dynamically-allocated shared memory size - a la Fortran-style
  // workspace.  Maximum shared memory size on Tesla is 0xc000.
  //
  size_t workBytes = sizeof(double) * (lx + rDim*(lx + rDim + 2));

  while (curIdx < pLength) {
    int chunkLength = min(pLength - curIdx, maxGrid);
    int chunkBytes = rDim * chunkLength * sizeof(double);
    LSFit2Ker<<<chunkLength,lx,workBytes>>>(dDesign, dY, &dyIdx[curIdx], &dp1[curIdx], &dp2[curIdx], dCoefs, dTscores, lx, rDim);
    cudaMemcpy(tScores + curIdx * rDim, dTscores, chunkBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(coefs + curIdx * rDim, dCoefs, chunkBytes, cudaMemcpyDeviceToHost);
    curIdx += chunkLength;
  }

  //  if (cublasGetError() != CUBLAS_STATUS_SUCCESS)
  //  cout << "WideLM:  cublas internal error" << endl;

  cublasFree(dDesign);
  cublasFree(dY);
  cudaFree(dyIdx);
  cudaFree(dp1);
  cudaFree(dp2);
  cudaFree(dCoefs);
  cudaFree(dTscores);
  cublasShutdown();
}

// Fits saturated and additive models on <= two predictors.
// Returns 't'-statistics and coefficients of fit.
//
// 'r' must be initialized to zero at each kernel invocation.
// 'x' is initialized to columns 'p1' and 'p2' of the design matrix,
//  followed by the slotwise product of these two columns.
// 'yx' must be initialized to the response vector.
// 'w' and 'coef' may be uninitialized.
//
extern __shared__ double smem[];
__global__ void LSFit2Ker(double *dDesign, double *dY, int *dyIdx, int *dp1, int *dp2, double *dCoefs, double *dTscores, int lx, int rDim) {

  int ncolx = rDim - 1;

  // These offsets and sizes should conform with 'workBytes' definition.
  //
  double *x = smem;
  double *yx = x + lx * ncolx;
  double *w = yx + lx;
  double *coef = w + rDim;
  double *r = coef + rDim;
  double *sdata = r + rDim * rDim; // sdata[lx];  Reduction workspace.
  //   double *test = sdata + lx;

  int xCol1Offs = dp1[blockIdx.x]* lx;
  int yColOffs = dyIdx[blockIdx.x] * lx;
  int xCol2Offs;
  if (ncolx >= 2)
    xCol2Offs = dp2[blockIdx.x]* lx;


  int tIdx = threadIdx.x;

  int j,k;
  double x1, x2;
  //  dcopy11(dDesign + lx*xCol1Offs, x);
  x1 = dDesign[xCol1Offs + tIdx];
  x[tIdx] = x1;

  //dcopy11(dDesign + lx*xCol2Offs, x+lx);
  if (ncolx >= 2) {
    x2 = dDesign[xCol2Offs + tIdx];
    x[lx + tIdx] = x2;
  }

  // Intersection term requested.
  //  dmul11(x, x+lx, x + 2*lx);
  if (ncolx == 3)
    x[2*lx + tIdx] = x1 * x2;

  //  dset(rDim * rDim, 0.0, r);
  if (tIdx < rDim * rDim)
    r[tIdx] = 0.0;

  //  dgemvt1a00(sdata, ncolx, dv2, x, lx, &w[1]);
  for (k = 0; k < ncolx; k++) {
    (void) dcusum(x + k*lx, sdata);
    if (tIdx == 0)
      w[k+1] = dv2 * sdata[0];
  }
  if (tIdx == 0) {
    w[0] = dw0;
    //    __threadfence(); // All threads to read 'w'.
  }
  __syncthreads(); // Why doesn't above 'threadfence' suffice?

  //  daxpy1y(rDim, -2.0*dv1, w, r, rDim);
  if (tIdx < rDim)
    r[rDim * tIdx] = -2.0*dv1 * w[tIdx];

  //dger01(lx, ncolx, -2.0 * dv2, &w[1], x, lx);
  for (k = 0; k < ncolx; k++)
    if (tIdx < dnrow)
      x[lx*k + tIdx] += -2.0 *dv2 * w[k+1];

  //  int colOffset = 0;
  int ncols = ncolx;
  double *rDiag = r + rDim + 1;
  for (j = 0; j < rDim-1; j++) {
    double vn2 = dcunrm2(x + j*lx, sdata);  // sync side-effect
    int v0 = vn2 > 0.0 ? 0 : 1;
    double recip  = 1.0 / (sqrt(vn2) * !v0 + v0);
    double alpha = dv1 * (recip - v0);

    // dgemvt1a01(sdata, ncols, alpha, x + colOffset, lx, x + colOffset, w);
    for (k = 0; k < ncols; k++) {
      (void) dcudot11(x + (j+k)*lx, x + j*lx, sdata);
      if (tIdx == 0)
	w[k] = alpha * sdata[0];
    }
    __syncthreads();  // All threads read 'w', set by thread 0.

    // daxpy1x(ncols, dv1, rDiag, rDim, w);
    // daxpy1y(ncols, -2.0*dv1, w, rDiag, rDim);
    if (tIdx < ncols)
      rDiag[rDim*tIdx] = -2.0 *dv1 * w[tIdx];
 
    //    dscal1(alpha, x+colOffset);
    x[j*lx + tIdx] *= alpha;

    //  dger11(lx, ncols-1, -2.0, x+colOffset, w+1, x+colOffset+lx, lx);
    for (k = 0; k < ncols -1; k++) {
      x[(j+k+1)*lx + tIdx] += w[k+1] * -2.0 * x[j * lx + tIdx];
    }

    rDiag += rDim + 1;
    ncols--;
  }
  // Current value of w[] is dead.
  //

  //  dcopy11(dY[, dyIdx], yx);
  yx[tIdx] = dY[yColOffs + tIdx];
  double temp = -2.0 * dv2 * dcusum(yx, sdata);
  if (tIdx == 0)
    coef[0] = temp*dv1;

  //  daxpy01(dv2 * temp, yx);
  if (tIdx < dnrow)
    yx[tIdx] += dv2*temp;

  for (j = 0; j < rDim-1; j++) {
    temp = -2.0 * dcudot11(x+ j*lx, yx, sdata);
    if (tIdx == 0)
      coef[j+1] = temp * dv1; // Next use takes place in thread 0.

    //    daxpy11(temp, x+colOffset, yx);
    if (tIdx < dnrow)
      yx[tIdx] += x[tIdx + j*lx] * temp;
  }
  if (tIdx == 0) {
    dtrsvUNN1(rDim, r, rDim, coef);
  }

  (void) dcunrm2(yx, sdata);// RSS: * recipDF;

  for (j = 0; j < rDim; j++) {
    // (void) dset(rDim, 0.0, yx):  can zero out entire vector.
    if (tIdx == j)
      yx[tIdx] = 1.0;
    else
      yx[tIdx] = 0.0;
    __syncthreads(); // Thread 0 reads 'yx'.
    if (tIdx == 0) {
      dtrsvUTN1(rDim, r, rDim, yx);
      dtrsvUNN1(rDim, r, rDim, yx);
      w[j] = sqrt(sdata[0] * yx[j] * dRecipDF);
    }
  }

  __syncthreads(); // Reads 'w' and 'coef'.
  if (tIdx < rDim) {
    dCoefs[blockIdx.x * rDim + tIdx] = coef[tIdx];
    dTscores[blockIdx.x * rDim + tIdx] = coef[tIdx] / w[tIdx];
  }
}


