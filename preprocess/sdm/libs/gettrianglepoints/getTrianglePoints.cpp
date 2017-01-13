#include <math.h>
#include <matrix.h>
#include <mex.h>

typedef char            int8;
typedef short           int16;
typedef long            int32;

mxArray *interpSingleToSingle(size_t nInst, float *mP1, float *mP2, float *mP3, float *mC1, float *mC2){
    mxArray *ret = mxCreateNumericMatrix(nInst, 2, mxSINGLE_CLASS, mxREAL);
    float *mat = (float *)mxGetPr(ret);
    
    for(size_t i=0;i<nInst;i++){
        size_t j = i+nInst;
        
        float tx = mP1[i] + (mP2[i] - mP1[i]) * mC1[i];
        float ty = mP1[j] + (mP2[j] - mP1[j]) * mC1[i];
        mat[i] = mP3[i] + (tx - mP3[i]) * mC2[i];
        mat[j] = mP3[j] + (ty - mP3[j]) * mC2[i];
    }
    
    return ret;
}

mxArray *interpSingleToDouble(size_t nInst, float *mP1, float *mP2, float *mP3, double *mC1, double *mC2){
    mxArray *ret = mxCreateNumericMatrix(nInst, 2, mxDOUBLE_CLASS, mxREAL);
    double *mat = (double *)mxGetPr(ret);
    
    for(size_t i=0;i<nInst;i++){
        size_t j = i+nInst;
        
        double tx = mP1[i] + (mP2[i] - mP1[i]) * mC1[i];
        double ty = mP1[j] + (mP2[j] - mP1[j]) * mC1[i];
        mat[i] = mP3[i] + (tx - mP3[i]) * mC2[i];
        mat[j] = mP3[j] + (ty - mP3[j]) * mC2[i];
    }
    
    return ret;
}

mxArray *interpDoubleToSingle(size_t nInst, double *mP1, double *mP2, double *mP3, float *mC1, float *mC2){
    mxArray *ret = mxCreateNumericMatrix(nInst, 2, mxSINGLE_CLASS, mxREAL);
    float *mat = (float *)mxGetPr(ret);
    
    for(size_t i=0;i<nInst;i++){
        size_t j = i+nInst;
        
        double tx = mP1[i] + (mP2[i] - mP1[i]) * mC1[i];
        double ty = mP1[j] + (mP2[j] - mP1[j]) * mC1[i];
        mat[i] = (float) (mP3[i] + (tx - mP3[i]) * mC2[i]);
        mat[j] = (float) (mP3[j] + (ty - mP3[j]) * mC2[i]);
    }
    
    return ret;
}

mxArray *interpDoubleToDouble(size_t nInst, double *mP1, double *mP2, double *mP3, double *mC1, double *mC2){
    mxArray *ret = mxCreateNumericMatrix(nInst, 2, mxDOUBLE_CLASS, mxREAL);
    double *mat = (double *)mxGetPr(ret);
    
    for(size_t i=0;i<nInst;i++){
        size_t j = i+nInst;
        
        double tx = mP1[i] + (mP2[i] - mP1[i]) * mC1[i];
        double ty = mP1[j] + (mP2[j] - mP1[j]) * mC1[i];
        mat[i] = mP3[i] + (tx - mP3[i]) * mC2[i];
        mat[j] = mP3[j] + (ty - mP3[j]) * mC2[i];
    }
    
    return ret;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    if(nrhs != 5) mexErrMsgTxt("Invalid number of right hand side parameters.");
    
    // Check first parameter
    size_t *ndim1 = (size_t *)mxGetDimensions(prhs[0]);
    if(mxGetNumberOfDimensions(prhs[0]) != 2 || ndim1[1] != 2){
        mexErrMsgTxt("First parameter must be of size Nx2.");
    }
    
    // Check second parameter
    size_t *ndim2 = (size_t *)mxGetDimensions(prhs[1]);
    if(mxGetNumberOfDimensions(prhs[1]) != 2 || ndim2[1] != 2){
        mexErrMsgTxt("Second parameter must be of size Nx2.");
    }
    
    // Check third parameter
    size_t *ndim3 = (size_t *)mxGetDimensions(prhs[2]);
    if(mxGetNumberOfDimensions(prhs[2]) != 2 || ndim3[1] != 2){
        mexErrMsgTxt("Third parameter must be of size Nx2.");
    }
    
    // Check fourth parameter
    size_t *ndim4 = (size_t *)mxGetDimensions(prhs[3]);
    if(mxGetNumberOfDimensions(prhs[3]) != 2 || ndim4[1] != 1){
        mexErrMsgTxt("Fourth parameter must be of size Nx1");
    }
    
    // Check fifth parameter
    size_t *ndim5 = (size_t *)mxGetDimensions(prhs[4]);
    if(mxGetNumberOfDimensions(prhs[4]) != 2 || ndim5[1] != 1){
        mexErrMsgTxt("Fifth parameter must be of size Nx1");
    }
    
    // Check parameter dimensions are consistent
    if(ndim1[0] != ndim2[0] || ndim1[0] != ndim3[0] || ndim1[0] != ndim4[0] || ndim1[0] != ndim5[0]){
        mexErrMsgTxt("All parameters must have the same number of rows.");
    }
    
    // Get parameter classes
    mxClassID cP1 = mxGetClassID(prhs[0]);
    mxClassID cP2 = mxGetClassID(prhs[1]);
    mxClassID cP3 = mxGetClassID(prhs[2]);
    mxClassID cP4 = mxGetClassID(prhs[3]);
    mxClassID cP5 = mxGetClassID(prhs[4]);
    
    // Check types for the first three parameters are consistent
    if((cP1 != cP2) || (cP1 != cP3)){
        mexErrMsgTxt("The three first parameters must be of the same class.");
    }
    
    // Check types for the two last parameters are consistent
    if((cP4 != cP5) || (cP4 != mxDOUBLE_CLASS && cP4 != mxSINGLE_CLASS)){
        mexErrMsgTxt("Fourth and fifth parameters must either be single or double.");
    }
    
    // Capture matrix data pointers
    void *mP1 = (void *)mxGetPr(prhs[0]);
    void *mP2 = (void *)mxGetPr(prhs[1]);
    void *mP3 = (void *)mxGetPr(prhs[2]);
    void *mC1 = (void *)mxGetPr(prhs[3]);
    void *mC2 = (void *)mxGetPr(prhs[4]);
    
    
    switch(cP4){
        case mxSINGLE_CLASS:
            switch(cP1){
                case mxSINGLE_CLASS:    plhs[0] = interpSingleToSingle(ndim1[0], (float *)mP1,  (float *)mP2,  (float *)mP3,  (float *)mC1,  (float *)mC2);   break;
                case mxDOUBLE_CLASS:    plhs[0] = interpDoubleToSingle(ndim1[0], (double *)mP1, (double *)mP2, (double *)mP3, (float *)mC1,  (float *)mC2);   break;
            }
            break;
            
        case mxDOUBLE_CLASS:
            switch(cP1){
                case mxSINGLE_CLASS:    plhs[0] = interpSingleToDouble(ndim1[0], (float *)mP1,  (float *)mP2,  (float *)mP3,  (double *)mC1, (double *)mC2);  break;
                case mxDOUBLE_CLASS:    plhs[0] = interpDoubleToDouble(ndim1[0], (double *)mP1, (double *)mP2, (double *)mP3, (double *)mC1, (double *)mC2);  break;
            }
            break;
            
    }
}