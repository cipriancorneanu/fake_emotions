#include <math.h>
#include <matrix.h>
#include <mex.h>
#include <float.h>

typedef short		int16;
typedef long		int32;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    if(nrhs != 3) mexErrMsgTxt("The function requires three right hand side parameters.");
    
    // Check input argument types
    mxClassID cBins       = mxGetClassID(prhs[0]);
    mxClassID cThresholds = mxGetClassID(prhs[1]);
    mxClassID cInstances  = mxGetClassID(prhs[2]);
    if(cBins != mxSINGLE_CLASS || cThresholds != mxSINGLE_CLASS || cInstances != mxSINGLE_CLASS){
        mexErrMsgTxt("Bins, thresholds and instances must be single valued.");
    }
    
    // Get input argument sizes
    size_t *dBins         = (size_t *)mxGetDimensions(prhs[0]);
    size_t *dThresholds   = (size_t *)mxGetDimensions(prhs[1]);
    size_t *dInstances    = (size_t *)mxGetDimensions(prhs[2]);
    
    // Define some variables
    size_t nTrsh = dThresholds[0]*dThresholds[1];
    size_t nInst = dInstances[0];
    size_t nBins = dBins[0];
    size_t nTarg = dBins[1];
    
    // Check thresholds are consistent
    if(dThresholds[0] > 1 && dThresholds[1] > 1){
        mexErrMsgTxt("Thresholds must be a row or column vector.");
    }
    
    // Check there is a bin for each threshold combination
    if(nBins != (size_t)pow((double)2,(double)nTrsh)){
        mexErrMsgTxt("Number of rows for the bins must 2^(number of threshold values).");
    }
    
    // Check input instances have as many columns as threshold values
    if(dInstances[1] != nTrsh){
        mexErrMsgTxt("Input instances must have a column for each threshold value.");
    }
    
    // Get matrix pointers
    float *mBins = (float *)mxGetPr(prhs[0]);
    float *mTrsh = (float *)mxGetPr(prhs[1]);
    float *mInst = (float *)mxGetPr(prhs[2]);
    
    // Prepare output matrix
    plhs[0] = mxCreateNumericMatrix(nInst, nTarg, mxSINGLE_CLASS, mxREAL);
    float *mat = (float *)mxGetPr(plhs[0]);
    
    // Calculate outputs
    for(int iI=0;iI<nInst;iI++){
        int index = 0;
        for(int j=0;j<nTrsh;j++){
            index |= ((mInst[iI+j*nInst] > mTrsh[j]) << j);
        }
        
        for(int j=0;j<nTarg;j++){
            mat[iI+j*nInst] = mBins[index+j*nBins];
        }
    }
}