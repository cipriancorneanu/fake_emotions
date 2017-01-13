#include <math.h>
#include <matrix.h>
#include <mex.h>

typedef short		int16;
typedef long		int32;

mxArray *executeRowsSingleInt16(float *data, int16 *indexs, size_t dataRows, size_t dataCols, double shrinkPar){
	// Get maximum index value
	int16 maxValue = 0;
	for(int i=0;i<dataRows;i++){
		if(indexs[i] > maxValue) maxValue = indexs[i];
	}

	// Create output matrix
    mxArray *ret = mxCreateNumericMatrix(maxValue, dataCols, mxSINGLE_CLASS, mxREAL);
	float *mat = (float *)mxGetPr(ret);

	// Initialize elements count
	size_t *counts = new size_t[maxValue];
	for(int16 i=0;i<maxValue;i++){
		counts[i] = 0;
	}

	// Sum values (count number of elements)
	for(size_t i=0;i<dataRows;i++){
		size_t index = indexs[i]-1;
		counts[index] += 1;
		for(size_t j=0;j<dataCols;j++) mat[index+j*maxValue] += data[i+j*dataRows];
	}

    // Average sums
    for(int16 i=0;i<maxValue;i++){
        for(size_t j=0;j<dataCols;j++){
            if(counts[i] > 0) mat[i+j*maxValue] /= (counts[i] + shrinkPar);
        }
    }

	delete[] counts;
	return ret;
}

mxArray *executeRowsSingleInt32(float *data, int32 *indexs, size_t dataRows, size_t dataCols, double shrinkPar){
	// Get maximum index value
	int32 maxValue = 0;
	for(int i=0;i<dataRows;i++){
		if(indexs[i] > maxValue) maxValue = indexs[i];
	}

	// Create output matrix
    mxArray *ret = mxCreateNumericMatrix(maxValue, dataCols, mxSINGLE_CLASS, mxREAL);
	float *mat = (float *)mxGetPr(ret);

	// Initialize elements count
	size_t *counts = new size_t[maxValue];
	for(int32 i=0;i<maxValue;i++){
		counts[i] = 0;
	}

	// Sum values (count number of elements)
	for(size_t i=0;i<dataRows;i++){
		size_t index = indexs[i]-1;
		counts[index] += 1;
		for(size_t j=0;j<dataCols;j++) mat[index+j*maxValue] += data[i+j*dataRows];
	}

    // Average sums
    for(int32 i=0;i<maxValue;i++){
        for(size_t j=0;j<dataCols;j++){
            if(counts[i] > 0) mat[i+j*maxValue] /= (counts[i] + shrinkPar);
        }
    }

	delete[] counts;
	return ret;
}

mxArray *executeRowsDoubleInt16(double *data, int16 *indexs, size_t dataRows, size_t dataCols, double shrinkPar){
	// Get maximum index value
	size_t maxValue = 0;
	for(size_t i=0;i<dataRows;i++){
		if(indexs[i] > maxValue) maxValue = indexs[i];
	}

	// Create output matrix
    mxArray *ret = mxCreateNumericMatrix(maxValue, dataCols, mxDOUBLE_CLASS, mxREAL);
	double *mat = (double *)mxGetPr(ret);

	// Initialize elements count
	size_t *counts = new size_t[maxValue];
	for(int16 i=0;i<maxValue;i++){
		counts[i] = 0;
	}

	// Sum values (count number of elements)
	for(size_t i=0;i<dataRows;i++){
		size_t index = indexs[i]-1;
		counts[index] += 1;
		for(size_t j=0;j<dataCols;j++) mat[index+j*maxValue] += data[i+j*dataRows];
	}

    // Average sums
    for(int16 i=0;i<maxValue;i++){
        for(size_t j=0;j<dataCols;j++){
            if(counts[i] > 0) mat[i+j*maxValue] /= (counts[i] + shrinkPar);
        }
    }

	delete[] counts;
	return ret;
}

mxArray *executeRowsDoubleInt32(double *data, int32 *indexs, size_t dataRows, size_t dataCols, double shrinkPar){
	// Get maximum index value
	int32 maxValue = 0;
	for(size_t i=0;i<dataRows;i++){
		if(indexs[i] > maxValue) maxValue = indexs[i];
	}

	// Create output matrix
    mxArray *ret = mxCreateNumericMatrix(maxValue, dataCols, mxDOUBLE_CLASS, mxREAL);
	double *mat = (double *)mxGetPr(ret);

	// Initialize elements count
	size_t *counts = new size_t[maxValue];
	for(int32 i=0;i<maxValue;i++){
		counts[i] = 0;
	}

	// Sum values (count number of elements)
	for(size_t i=0;i<dataRows;i++){
		size_t index = indexs[i]-1;
		counts[index] += 1;
		for(size_t j=0;j<dataCols;j++) mat[index+j*maxValue] += data[i+j*dataRows];
	}

    // Average sums
    for(int32 i=0;i<maxValue;i++){
        for(size_t j=0;j<dataCols;j++){
            if(counts[i] > 0) mat[i+j*maxValue] /= (counts[i] + shrinkPar);
        }
    }

	delete[] counts;
	return ret;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    const mxArray *aValues, *aIndexs, *aShrink;
    const size_t *dimsIndexs, *dimsValues;
    size_t ndimIndexs, ndimValues;

    // Get values and indexs matrices
    aValues = prhs[0];
    aIndexs = prhs[1];
    aShrink = prhs[2];

    // Get values dimensions
    dimsValues = (size_t *)mxGetDimensions(aValues);
    ndimValues = mxGetNumberOfDimensions(aValues);
    
    // Get indexs dimensions
    dimsIndexs = (size_t *)mxGetDimensions(aIndexs);
    ndimIndexs = mxGetNumberOfDimensions(aIndexs);

	// Check dimension sizes are consistent
	if(ndimValues > 2 || ndimIndexs > ndimValues){
		mexErrMsgTxt("Too many dimensions for the first parameter.");
		return;
	}

	if(ndimIndexs > ndimValues){
		mexErrMsgTxt("Indexs dimensionality cannot exceed that of the data.");
		return;
	}

	if(ndimIndexs == 2 && dimsIndexs[0] > 1 && dimsIndexs[1] > 1){
		mexErrMsgTxt("One of the index dimensions must be unitary.");
		return;
	}

	if(dimsIndexs[0] > 1 && dimsIndexs[0] != dimsValues[0]){
		mexErrMsgTxt("The first dimension size must be the same for both data and indexs.");
		return;
	}

	if(dimsIndexs[1] > 1 && dimsIndexs[1] != dimsValues[1]){
		mexErrMsgTxt("The second dimension size must be the same for both data and indexs.");
		return;
	}
    
	// Get pointers to matrices
	void *values = mxGetPr(aValues);
	void *indexs = mxGetPr(aIndexs);

	// Check if averaging is performed
    double shrinkage = 0;
    if(nrhs > 2) shrinkage = *(double *)mxGetPr(aShrink);

    // Select specific merger depending on data types
	switch(mxGetClassID(aValues)){
		case mxDOUBLE_CLASS:
			switch(mxGetClassID(aIndexs)){
				case mxINT16_CLASS:
					plhs[0] = executeRowsDoubleInt16((double *)values, (int16 *)indexs, dimsValues[0], dimsValues[1], shrinkage);
					return;

				case mxINT32_CLASS:
					plhs[0] = executeRowsDoubleInt32((double *)values, (int32 *)indexs, dimsValues[0], dimsValues[1], shrinkage);
					return;

                default:
                    mexErrMsgTxt("First argument must be a matrix of singles or doubles.");
                    return;
                    
			}
			break;

		case mxSINGLE_CLASS:
			switch(mxGetClassID(aIndexs)){
				case mxINT16_CLASS:
					plhs[0] = executeRowsSingleInt16((float *)values, (int16 *)indexs, dimsValues[0], dimsValues[1], shrinkage);
					return;

				case mxINT32_CLASS:
					plhs[0] = executeRowsSingleInt32((float *)values, (int32 *)indexs, dimsValues[0], dimsValues[1], shrinkage);
					return;
                    
                default:
                    mexErrMsgTxt("First argument must be a matrix of single or double values.");
                    return;

			}
			break;
            
        default:
            mexErrMsgTxt("Second argument must be a matrix of int16 or int32 values.");
            break;;
	}
}