% Author: Arman Savran (arman.savran@boun.edu.tr)
% Date:   2008
% Outputs:
%   zmin      : minimum depth value
%   nrows     : subsampled number of rows
%   ncols     : subsampled number of columns
%   imfile    : image file name
%   data      : Nx5 matrix where columns are 3D coordinates and 2D
%   normalized image coordinates respectively. 2D coordinates are
%   normalized to the range [0,1]. N = nrows*ncols. In this matrix, values
%   that are equal to zmin denotes the background.


function [data, zmin, nrows, ncols, imfile] = read_bntfile(filepath)

%filepath = [directory '/' filename '.bnt'];


fid = fopen(filepath, 'r');
if (fid == -1)
    disp(['Could not open file, ' filepath]);
    return
end

nrows = fread(fid, 1, 'uint16');
ncols = fread(fid, 1, 'uint16');
zmin = fread(fid, 1, 'float64');

len = fread(fid, 1, 'uint16');
imfile = fread(fid, [1 len], 'uint8=>char');

% normally, size of data must be nrows*ncols*5
len = fread(fid, 1, 'uint32');
data = fread(fid, [len/5 5], 'float64');

fclose(fid);