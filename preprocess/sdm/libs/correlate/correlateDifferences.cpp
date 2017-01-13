#include <math.h>
#include <matrix.h>
#include <mex.h>

typedef short		int16;
typedef long		int32;

mxArray *correlatePearsonSingle(float *m1, float *m2, size_t nInst, size_t nR1, size_t nR2){
    mxArray *ret = mxCreateNumericMatrix(nR1*nR1, nR2, mxSINGLE_CLASS, mxREAL);
	float *mat = (float *)mxGetPr(ret);
    
    // Calculate variable means for first variable set
    float *means1 = (float *)new float[nR1];
    float *stdvs1 = (float *)new float[nR1];
    for(size_t iA=0;iA<nR1;iA++){
        float *r = &m1[nInst*iA];
        
        float mean = 0;
        for(size_t i=0;i<nInst;i++) mean += r[i];
        mean /= nInst;
        
        float stdv = 0;
        for(size_t i=0;i<nInst;i++){
            r[i] -= mean;
            stdv += r[i] * r[i];
        }
        stdv = sqrt(stdv);
        
        means1[iA] = mean;
        stdvs1[iA] = stdv;
    }
    
    // Calculate variable means for second variable set
    float *means2 = (float *)new float[nR2];
    float *stdvs2 = (float *)new float[nR2];
    for(size_t iB=0;iB<nR2;iB++){
        float *r = &m2[nInst*iB];
        
        float mean = 0;
        for(size_t i=0;i<nInst;i++) mean += r[i];
        mean /= nInst;
        
        float stdv = 0;
        for(size_t i=0;i<nInst;i++){
            r[i] -= mean;
            stdv += r[i] * r[i];
        }
        stdv = sqrt(stdv);
        
        means2[iB] = mean;
        stdvs2[iB] = stdv;
    }
    
    // Calculate correlation betweeen the two variable sets
	float *corrs = new float[nR1*nR2];
    for(size_t iA=0;iA<nR1;iA++){
        float *r1 = &m1[nInst*iA];
        
        for(size_t iB=0;iB<nR2;iB++){
            float *r2 = &m2[nInst*iB];

            float num = 0;
            for(size_t i=0;i<nInst;i++) num += r1[i] * r2[i];
            corrs[iA+iB*nR1] = num / (stdvs1[iA] * stdvs2[iB]);
        }
    }
    
    // Calculate correlations between differences for each pair of the first
    // variable set and the second variable set
    float sqr2 = (float)1.41421356237;
    for(size_t c1=0;c1<nR1;c1++){
        for(size_t c2=c1+1;c2<nR1;c2++){
            for(size_t iB=0;iB<nR2;iB++){
                float corr = (corrs[iB*nR1+c1] - corrs[iB*nR1+c2]) / sqr2;
                mat[(c1*nR1+c2)+iB*nR1*nR1] = corr;
                mat[(c2*nR1+c1)+iB*nR1*nR1] = -corr;
            }
        }
    }
    
    delete[] means1, stdvs2;
    delete[] means2, stdvs2;
    delete[] corrs;
    return ret;
}

mxArray *correlatePearsonDouble(double *m1, double *m2, size_t nInst, size_t nR1, size_t nR2){
    mxArray *ret = mxCreateNumericMatrix(nR1, nR2, mxDOUBLE_CLASS, mxREAL);
	double *mat = (double *)mxGetPr(ret);
    
    for(size_t iA=0;iA<nR1;iA++){
        double *r1 = &m1[nInst*iA];
        
        for(size_t iB=0;iB<nR2;iB++){
            double *r2 = &m2[nInst*iB];
            
            // Calculate variable means
            double mean1=0, mean2=0;
            for(size_t i=0;i<nInst;i++){
                mean1 += r1[i];
                mean2 += r2[i];
            }
            mean1 /= nInst;
            mean2 /= nInst;
            
            // Calculate correlation parts
            double num = 0;
            double denSq1 = 0;
            double denSq2 = 0;
            for(size_t i=0;i<nInst;i++){
                double t1 = (r1[i] - mean1);
                double t2 = (r2[i] - mean2);
                num += t1 * t2;
                denSq1 += t1 * t1;
                denSq2 += t2 * t2;
            }
            
            // Assign coefficient to output matrix
            mat[iA+iB*nR1] = num / (sqrt(denSq1) * sqrt(denSq2));
        }
    }
    
    return ret;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    
    // Check there are enough input values
    if(nrhs < 2) mexErrMsgTxt("Two single or double precision right hand side aruments of size NxA and NxB are expected.");
 
    // Check input values have the same number of rows and
    size_t *sizeVals1 = (size_t *)mxGetDimensions(prhs[0]);
    size_t *sizeVals2 = (size_t *)mxGetDimensions(prhs[1]);
    if(sizeVals1[0] != sizeVals2[0] || sizeVals1[0] == 1 || sizeVals1[1] == 1){
        mexErrMsgTxt("Both input matrices first dimension must be non-unitzary and of the same size.");
    }
    
    // Check input values have the same types
    if(mxGetClassID(prhs[0]) != mxGetClassID(prhs[1]) || (mxGetClassID(prhs[0]) != mxSINGLE_CLASS && mxGetClassID(prhs[0]) != mxDOUBLE_CLASS)){
        mexErrMsgTxt("Both input matrices must be of the same type, which must either be single or double precision reals.");
    }
    
    // Perform correlation
    if(mxGetClassID(prhs[0]) == mxSINGLE_CLASS){
        plhs[0] = correlatePearsonSingle((float *)mxGetPr(prhs[0]), (float *)mxGetPr(prhs[1]), sizeVals1[0], sizeVals1[1], sizeVals2[1]);
    }else{
        plhs[0] = correlatePearsonDouble((double *)mxGetPr(prhs[0]), (double *)mxGetPr(prhs[1]), sizeVals1[0], sizeVals1[1], sizeVals2[1]);
    }
}