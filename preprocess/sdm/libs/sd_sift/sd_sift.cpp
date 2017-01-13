#include <math.h>
#include <string.h>
#include <matrix.h>
#include <mex.h>

typedef short		int16;
typedef long		int32;

#define round(x)    (x<0 ? ceil((x)-0.5) : floor((x)+0.5))

const float ATAN_LB    = -1.5708f;
const float ATAN_UB    = +1.5708f;
const float ATAN_OSIZE = (ATAN_UB - ATAN_LB) / 36.0f;
const float ATAN_DSIZE = (ATAN_UB - ATAN_LB) / 8.0f;

struct RefMatrix{
    float dX;
    float dY;
    float gaus;
};

int          descSize;   // Size of a single SIFT descriptor
int          winSize;    // Height/width of the descriptor window
int          binSize;    // Height/width of each bin
float        step;       // Spacing between consecutive samples
int          ndSamples;  // Number of vertical/horizontal samples
int          nSamples;   // Total number of samples (ndSamples^2)
int          nsb;        // Number of vertical/horizontal spatial bins

int          imHeight;   // Image height
int          imWidth;    // Image width
double*      image;      // Grayscale image

RefMatrix*   matRef;     // Reference matrix for the relative sample pixel positions
double*      matSift;    // Matrix for the specific SIFT descriptor calculations

void describeLandmark(float x, float y, double *ret){
    double* timg;
    int idx;
    
    // Calculate gaussian blur value for each sample
    for(int i=0;i<nSamples;i++){
        int tx = x + matRef[i].dX;
        if(tx <= 0 || tx >= imHeight-1){
            matSift[i] = 0;
            continue;
        }
        
        int ty = y + matRef[i].dY;
        if(ty <= 0 || ty >= imWidth-1){
            matSift[i] = 0;
            continue;
        }
        
        timg = &image[tx*imWidth+ty];
        matSift[i] = 0.0967*timg[-imWidth-1]  +  0.1176*timg[-imWidth]  +  0.0967*timg[-imWidth+1]  +
                     0.1176*timg[-1]          +  0.1429*timg[0]         +  0.1176*timg[+1]          +
                     0.0967*timg[+imWidth+1]  +  0.1176*timg[+imWidth]  +  0.0967*timg[+imWidth+1]  ;
    }
    
    // Calculate gradient orientations, magnitudes and generate orientations histogram
    idx = 0;
    double histOrient[36] = {0};
    for(int i=0;i<ndSamples;i++){
        for(int j=0;j<ndSamples;j++){
            double xl = (i > 0)           ? matSift[idx - ndSamples] : 0;
            double xu = (i < ndSamples-1) ? matSift[idx + ndSamples] : 0;
            double yl = (j > 0)           ? matSift[idx - 1] : 0;
            double yu = (j < ndSamples-1) ? matSift[idx + 1] : 0;
            
            double dx = xu-xl;
            double dy = yu-yl;
            
            int histOIdx = (dx>0) ? (atan(dy/dx)-ATAN_LB)/ATAN_OSIZE-0.5f : (dy>0)*35;
            histOrient[histOIdx] += sqrt(dx*dx + dy*dy) * matRef[idx].gaus;
            
            idx++;
        }
    }
    
    // Get orientation
    double rangle=0, maxVal=0;
    for(int i=0;i<36;i++){
        if(histOrient[i] > maxVal){
            maxVal = histOrient[i];
            rangle = (0.5+i) * ATAN_OSIZE;
        }
    }
    
    // Calculate rotated grid intensities
    double rcos = cos(rangle);
    double rsin = sin(rangle);
    for(int i=0;i<nSamples;i++){
        int tx = x + matRef[i].dX * rcos - matRef[i].dY * rsin;
        if(tx<=0 || tx >= imHeight-1){
            matSift[i] = 0;
            continue;
        }
        
        int ty = y + matRef[i].dY * rcos + matRef[i].dX * rsin;
        if(ty<=0 || ty >= imWidth-1){
            matSift[i] = 0;
            continue;
        }

        timg = &image[tx*imWidth+ty];
        matSift[i] = 0.0967*timg[-imWidth-1]  +  0.1176*timg[-imWidth]  +  0.0967*timg[-imWidth+1]  +
                     0.1176*timg[-1]          +  0.1429*timg[0]         +  0.1176*timg[+1]          +
                     0.0967*timg[+imWidth+1]  +  0.1176*timg[+imWidth]  +  0.0967*timg[+imWidth+1]  ;
    }
    
    // Calculate resulting bins histogram
    idx = 0;
    for(int i=0;i<ndSamples;i++){
        int binX = i / binSize;
        
        for(int j=0;j<ndSamples;j++){
            int binY = j / binSize;
            
            double xl = (i > 0)           ? matSift[idx - ndSamples] : 0;
            double xu = (i < ndSamples-1) ? matSift[idx + ndSamples] : 0;
            double yl = (j > 0)           ? matSift[idx - 1] : 0;
            double yu = (j < ndSamples-1) ? matSift[idx + 1] : 0;
            
            double dx = xu-xl;
            double dy = yu-yl;

            int histDIdx = (dx>0) ? (atan(dy/dx)-ATAN_LB)/ATAN_DSIZE-0.5f : (dy>0)*7;
            ret[(binX*nsb+binY)*8 + histDIdx] += sqrt(dx*dx + dy*dy) * matRef[idx].gaus;
            
            idx++;
        }
    }
    
    // Normalize histogram
    for(int i=0;i<descSize;i+=8){
        double *tret = &ret[i];
        double sum = tret[0] + tret[1] + tret[2] + tret[3] + tret[4] + tret[5] + tret[6] + tret[7];
        if(sum > 0){
            for(int j=0;j<8;j++) tret[j] /= sum;
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    int *tsizes;
    
    // Check there are enough input values
    if(nrhs < 2) mexErrMsgTxt("The function requires at least two parameters: an input image and an Nx2 double precision array indicating the locations at which SIFT descriptors must be computed.");
    
    // Read input image
    tsizes = (int *)mxGetDimensions(prhs[0]);
    if(mxGetClassID(prhs[0]) != mxDOUBLE_CLASS || mxGetNumberOfDimensions(prhs[0]) > 2) mexErrMsgTxt("Input image must be a grayscale image of double precision.");
    image    = (double *) mxGetPr(prhs[0]);
    imHeight = (int) tsizes[0];
    imWidth  = (int) tsizes[1];
    
    // Read input list of landmarks
    tsizes = (int *)mxGetDimensions(prhs[1]);
    if(mxGetClassID(prhs[1]) != mxDOUBLE_CLASS || mxGetNumberOfDimensions(prhs[1]) > 2 || tsizes[1] != 2) mexErrMsgTxt("List of landmarks must be an Nx2 double precision array.");
    double *landmarks = (double *) mxGetPr(prhs[1]);
    int numLandmarks  = (int) tsizes[0];
    
    // Initialize parameters
    winSize  = 32;
    nsb      = 4;
    binSize  = winSize / nsb;
    
    // Read optional parameters
    for(int i=2;i<nrhs;i+=2){
        if(mxGetClassID(prhs[i]) != mxCHAR_CLASS) mexErrMsgTxt("Optional argument value passed without specifying the parameter name.");
        if(nrhs-1 == i) mexErrMsgTxt("Optional argument name specified but no value has been assigned.");
        char *str = mxArrayToString(prhs[i]);
        
        if(strcmp("nsb\n", str) == 0){
            tsizes = (int *)mxGetDimensions(prhs[i+1]);
            if(mxGetNumberOfDimensions(prhs[i+1]) > 2 || tsizes[0] > 1 || tsizes[1] > 1){
                mexErrMsgTxt("The number of spatial bins must be a scalar value.");
            }
            
            if(mxGetClassID(prhs[i+1]) == mxSINGLE_CLASS) nsb = (int) round(*(float *) mxGetPr(prhs[i+1]));
            if(mxGetClassID(prhs[i+1]) == mxDOUBLE_CLASS) nsb = (int) round(*(double *) mxGetPr(prhs[i+1]));
            binSize = round((float)winSize / (float)nsb);
            
        }else if(strcmp("winsize\n", str)){
            tsizes = (int *)mxGetDimensions(prhs[i+1]);
            if(mxGetNumberOfDimensions(prhs[i+1]) > 2 || tsizes[0] > 1 || tsizes[1] > 1){
                mexErrMsgTxt("The window size must be a scalar value.");
            }
            
            if(mxGetClassID(prhs[i+1]) == mxSINGLE_CLASS) winSize = (int) round(*(float *) mxGetPr(prhs[i+1]));
            if(mxGetClassID(prhs[i+1]) == mxDOUBLE_CLASS) winSize = (int) round(*(double *) mxGetPr(prhs[i+1]));
            binSize = round((float)winSize / (float)nsb);
        }
    }
    
    // Update window size to have equally-sized bins
    winSize = binSize * nsb;

    // Calculate step between pixels and number of samples at each dimension
    step      = (winSize <= 32) ? 1 : winSize / 32.0f;
    ndSamples = (winSize < 32) ? winSize : 32;
    binSize   = ndSamples / nsb;
    nSamples  = ndSamples*ndSamples;
    
    // Define orientative array of samples
    matRef       = new RefMatrix[nSamples];
    matSift      = new double[nSamples];
    float tsigma = winSize / 2.0f;
    for(int i=0;i<ndSamples/2;i++){
        for(int j=0;j<ndSamples/2;j++){
            int idx1 = (ndSamples/2 - i -1)*ndSamples + (ndSamples/2 - j - 1);
            int idx2 = (ndSamples/2 + i)*ndSamples + (ndSamples/2 - j - 1);
            int idx3 = (ndSamples/2 + i)*ndSamples + (ndSamples/2 + j);
            int idx4 = (ndSamples/2 - i - 1)*ndSamples + (ndSamples/2 + j);
            
            float dX    = (i+1)*step;
            float dY    = (j+1)*step;
            float tdist = sqrt(dX*dX + dY*dY);
            float tgaus = 1.0f / (tsigma * 2.5066f) * exp(-0.5f * (tdist/tsigma)*(tdist/tsigma));
            
            matRef[idx1].dX   = -dX;
            matRef[idx1].dY   = -dY;
            matRef[idx1].gaus = tgaus;
            
            matRef[idx2].dX   = +dX;
            matRef[idx2].dY   = -dY;
            matRef[idx2].gaus = tgaus;
            
            matRef[idx3].dX   = +dX;
            matRef[idx3].dY   = +dY;
            matRef[idx3].gaus = tgaus;
            
            matRef[idx4].dX   = -dX;
            matRef[idx4].dY   = +dY;
            matRef[idx4].gaus = tgaus;
        }
    }
    
    // Prepare return matrix
    descSize    = 8*nsb*nsb;
    plhs[0]     = mxCreateNumericMatrix(descSize, numLandmarks, mxDOUBLE_CLASS, mxREAL);
    double *mat = (double *)mxGetPr(plhs[0]);
    
    // Create descriptor at each landmark location
    for(int i=0;i<numLandmarks;i++){
        describeLandmark((float)landmarks[i], (float)landmarks[i+numLandmarks], &mat[i*descSize]);
    }
    
    // Clear memory
    delete[] matRef;
    delete[] matSift;
}