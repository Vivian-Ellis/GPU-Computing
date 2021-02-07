#include <stdio.h>
#include "kernel1.h"


//extern  __shared__  float sdata[];

////////////////////////////////////////////////////////////////////////////////
//! Weighted Jacobi Iteration
//! @param g_dataA  input data in global memory
//! @param g_dataB  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void k1( float* g_dataA, float* g_dataB, int floatpitch, int width) 
{
  extern __shared__ float s_data[];

  // global thread(data) row index 
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  i = i + 1; //because the edge of the data is not processed
  
  // global thread(data) column index
  unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

  int sharedDim = blockDim.x+2;
  int ty = threadIdx.y+1;
  int tx = threadIdx.x+1;

  j = j + 1; //because the edge of the data is not processed  

  // check the boundary
  if( i >= width - 1|| j >= width - 1|| i < 1 || j < 1) return;

//we will only copy the contents first. there are three cases
//1 left boundary, 2 right boundary and 3 middle

     //case 1 middle
	s_data[ty * sharedDim + tx]=g_dataA[i * floatpitch + j]; //itself
	s_data[(ty-1) * sharedDim +  tx]= g_dataA[(i-1) * floatpitch +  j];//N
	s_data[(ty+1) *  sharedDim+  tx]=g_dataA[(i+1) * floatpitch +  j];//S

    //case 2 left boundary
        s_data[(ty-1) *  sharedDim+ (tx-1)]=g_dataA[(i-1) * floatpitch + (j-1)];//NW
        s_data[ ty    *  sharedDim+ (tx-1)]=g_dataA[ i    * floatpitch + (j-1)];//W
	s_data[(ty+1) *  sharedDim+ (tx-1)]=g_dataA[(i+1) * floatpitch + (j-1)];//SW
 
    //case 3 right boundary
        s_data[(ty-1) * sharedDim + (tx+1)] =g_dataA[(i-1) * floatpitch + (j+1)];//NE
        s_data[ ty    * sharedDim+ (tx+1)] =g_dataA[ i    * floatpitch + (j+1)];//E
        s_data[(ty+1) * sharedDim + (tx+1)] =g_dataA[(i+1) * floatpitch + (j+1)];//SE
 
  __syncthreads();//wait for all threads to finish

  g_dataB[i * floatpitch + j] = (
                0.2f*s_data[ty * sharedDim + tx] +               //itself
                0.1f*s_data[(ty-1) * sharedDim +  tx   ] +       //N
                0.1f*s_data[(ty-1) * sharedDim + (tx+1)] +       //NE
                0.1f*s_data[ ty    *  sharedDim+ (tx+1)] +       //E
                0.1f*s_data[(ty+1) * sharedDim + (tx+1)] +       //SE
                0.1f*s_data[(ty+1) *  sharedDim+  tx   ] +       //S
                0.1f*s_data[(ty+1) *  sharedDim+ (tx-1)] +       //SW
                0.1f*s_data[ ty    *  sharedDim+ (tx-1)] +       //W
                0.1f*s_data[(ty-1) *  sharedDim+ (tx-1)]         //NW
                ) *0.95f;
}//end kernel1

