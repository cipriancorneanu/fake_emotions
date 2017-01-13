#include <math.h>
#include <matrix.h>
#include <mex.h>
#include <float.h>

typedef short		int16;
typedef long		int32;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    if(nrhs != 4) mexErrMsgTxt("The function requires four right hand side parameters.");
    
    // Check input argument types
    mxClassID cTargs = mxGetClassID(prhs[0]);
    mxClassID cThres = mxGetClassID(prhs[1]);
    mxClassID cCombs = mxGetClassID(prhs[2]);
    mxClassID cShrink = mxGetClassID(prhs[3]);
    if(cTargs != mxSINGLE_CLASS || cThres != mxSINGLE_CLASS || cCombs != mxDOUBLE_CLASS || cShrink != mxDOUBLE_CLASS){
        mexErrMsgTxt("Targets and feature matrices must single valued.");
    }
    
    // Check input argument sizes
    size_t *dTargs = (size_t *)mxGetDimensions(prhs[0]);
    size_t *dThres = (size_t *)mxGetDimensions(prhs[1]);
    size_t *dCombs = (size_t *)mxGetDimensions(prhs[2]);
    size_t *dShrink = (size_t *)mxGetDimensions(prhs[3]);
    if(dThres[0] != dTargs[0]){
        mexErrMsgTxt("Number of rows must be the same for the two first arguments.");
    }
    
    // Obtain input arguments
    float * mTargs  = (float *)mxGetPr(prhs[0]);
    float * mThres  = (float *)mxGetPr(prhs[1]);
    size_t  nComb   = (size_t)*(double *)mxGetPr(prhs[2]);
    double  vShrink = *(double *)mxGetPr(prhs[3]);
    
    // Define some variables
    size_t nInst = dThres[0];
    size_t nVars = dThres[1];
    size_t nTarg = dTargs[1];
    size_t nBins = (size_t)pow((double)2,(double)nVars);
    
    // Obtain min/max threshold values
    float *minThrsh = new float[nVars];
    float *maxThrsh = new float[nVars];
    for(int i=0;i<dThres[1];i++){
        minThrsh[i] = mThres[i*nInst];
        maxThrsh[i] = minThrsh[i];
    }
    for(int i=1;i<nInst;i++){
        for(int j=0;j<nVars;j++){
            float cv = mThres[i+j*nInst];
            if(cv<minThrsh[j]) minThrsh[j] = cv;
            if(cv>maxThrsh[j]) maxThrsh[j] = cv;
        }
    }     
    
    // Allocate required variables
    int *      indexs      = new int[nInst];
    size_t *   counts      = new size_t[nBins];
    float *    thrsh       = new float[nComb*nVars];
    mxArray ** binMatrices = new mxArray*[nComb];
    for(size_t i=0;i<nComb;i++){
        binMatrices[i] = mxCreateNumericMatrix(nBins, nTarg, mxSINGLE_CLASS, mxREAL);
    }
    
    // Prepare best solution tracking
    float bestError = FLT_MAX;
    int bestRegressor = 0;
    
    // Try nCombs threshold combinations
    for(size_t iR=0;iR<nComb;iR++){
        float *binMatrix = (float *)mxGetPr(binMatrices[iR]);
        float *thresholds = &thrsh[iR*nVars];
        
        // Generate random thresholds
        for(int i=0;i<nVars;i++){
            thresholds[i] = minThrsh[i] + ((float)rand()/(float)RAND_MAX) * (maxThrsh[i] - minThrsh[i]);
        }
        
        // Initialize bin elements count
        for(size_t i=0;i<nBins;i++) counts[i] = 0;
        
        // Sum values (count number of elements)
        for(size_t i=0;i<nInst;i++){
            indexs[i] = 0;
            for(int j=0;j<nVars;j++){
                indexs[i] |= ((mThres[i+j*nInst] > thresholds[j]) << j);
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
        
        // Calculate error (L2 instances)
        float cerr = 0;
        for(size_t i=0;i<nInst;i++){
            float tcerr = 0;
            for(size_t j=0;j<nTarg;j++){
                float tval = binMatrix[indexs[i]+j*nBins] - mTargs[i+j*nInst];
                tcerr += tval*tval;
            }
            cerr += sqrt(tcerr);
        }
        cerr /= nInst;
        
        // If error lower, save current regressor as the best
        if(cerr<bestError){
            bestError = cerr;
            bestRegressor = iR;
        }
    }
    
    // Prepare returns
    plhs[0] = binMatrices[bestRegressor];
    plhs[1] = mxCreateNumericMatrix(1, nVars, mxSINGLE_CLASS, mxREAL);
    plhs[2] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    
    float *fthrs = (float *)mxGetPr(plhs[1]);
    for(size_t i=0;i<nVars;i++) fthrs[i] = thrsh[bestRegressor*nVars+i];
    
    float *ferr = (float *)mxGetPr(plhs[2]);
    ferr[0] = bestError;
}