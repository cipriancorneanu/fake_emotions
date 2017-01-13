#include <math.h>
#include <matrix.h>
#include <mex.h>
#include <float.h>

typedef short		int16;
typedef long		int32;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    if(nrhs != 4) mexErrMsgTxt("The function requires four right hand side parameters.");
    
    // Check input argument types
    mxClassID cTargs  = mxGetClassID(prhs[0]);
    mxClassID cFeats  = mxGetClassID(prhs[1]);
    mxClassID cThres  = mxGetClassID(prhs[2]);
    mxClassID cShrink = mxGetClassID(prhs[3]);
    if(cTargs != mxSINGLE_CLASS || cFeats != mxSINGLE_CLASS || cThres != mxSINGLE_CLASS || cShrink != mxDOUBLE_CLASS){
        mexErrMsgTxt("Target, feature and threshold matrices must single valued.");
    }
    
    // Check input argument sizes
    size_t *dTargs  = (size_t *)mxGetDimensions(prhs[0]);
    size_t *dFeats  = (size_t *)mxGetDimensions(prhs[1]);
    size_t *dThres  = (size_t *)mxGetDimensions(prhs[2]);
    size_t *dShrink = (size_t *)mxGetDimensions(prhs[2]);
    if(dTargs[0] != dFeats[0]){
        mexErrMsgTxt("Number of rows must be the same for the two first arguments.");
    }
    
    // Obtain input arguments
    float * mTargs  = (float *)mxGetPr(prhs[0]);
    float * mFeats  = (float *)mxGetPr(prhs[1]);
    float * mThres  = (float *)mxGetPr(prhs[2]);
    float   vShrink = (float)*(double *)mxGetPr(prhs[3]);
    
    // Define some variables
    size_t nInst = dTargs[0];
    size_t nVars = dThres[1];
    size_t nTarg = dTargs[1];
    size_t nBins = (size_t)pow((double)2,(double)nVars);
    
    // Allocate required variables
    int *      indexs      = new int[nInst];
    size_t *   counts      = new size_t[nBins];
    
    // Allocate output bins
    plhs[0] = mxCreateNumericMatrix(nBins, nTarg, mxSINGLE_CLASS, mxREAL);
    float *binMatrix = (float *)mxGetPr(plhs[0]);

    // Initialize bin elements count
    for(size_t i=0;i<nBins;i++) counts[i] = 0;

    // Sum values (count number of elements)
    for(size_t i=0;i<nInst;i++){
        indexs[i] = 0;
        for(int j=0;j<nVars;j++){
            indexs[i] |= ((mFeats[i+j*nInst] > mThres[j]) << j);
        }

        counts[indexs[i]] += 1;
        for(size_t j=0;j<nTarg;j++){
            binMatrix[indexs[i]+j*nBins] += mTargs[i+j*nInst];
        }
    }

    // Average sums
    for(size_t i=0;i<nBins;i++){
        for(size_t j=0;j<nTarg;j++){
            if(counts[i] > 0) binMatrix[i+j*nBins] /= (counts[i] + vShrink);
        }
    }

    // Prepare error return matrix
    plhs[1] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    float *err = (float *)mxGetPr(plhs[1]);
    
    // Calculate error (L2 instances)
    err[0] = 0;
    for(size_t i=0;i<nInst;i++){
        float terr = 0;
        for(size_t j=0;j<nTarg;j++){
            float tval = binMatrix[indexs[i]+j*nBins] - mTargs[i+j*nInst];
            terr += tval*tval;
        }
        err[0] += sqrt(terr);
    }
    err[0] /= nInst;
    
    delete[] indexs;
    delete[] counts;
}