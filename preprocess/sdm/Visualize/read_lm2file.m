% Author: Arman Savran (arman.savran@boun.edu.tr)
% Date:   2008

function [pts2D, labels] = read_lm2file(filepath)

fid = fopen(filepath, 'rt');
if (fid == -1)
    error(['Could not open file, ' filepath]);
end

fgetl(fid);
fgetl(fid);

N = fscanf(fid, '%d');
fgetl(fid);
fgetl(fid);
fgetl(fid);

labels = cell(N,1);
for n=1:N
    labels{n} = fgetl(fid);
end

fgetl(fid);
fgetl(fid);

pts2D = fscanf(fid, '%g %g', [2 inf]);

fclose(fid);