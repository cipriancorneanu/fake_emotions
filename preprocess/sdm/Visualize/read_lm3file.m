% Author: Arman Savran (arman.savran@boun.edu.tr)
% Date:   2008

function [pts3D, labels] = read_lm3file(filepath)

fid = fopen(filepath, 'rt');
if (fid == -1)
    error(['Could not open file, ' filepath]);
end

fgetl(fid);
if fgetl(fid) == '*'
    % here after in the file, some info text is written -> skip them
    while ( ~isempty(fgetl(fid)) )
    end
end

N = fscanf(fid, '%d');
labels = cell(N,1);
pts3D = zeros(3,N);

fgetl(fid);
for n=1:N
    labels{n} = fgetl(fid);
    pts3D(:,n) = fscanf(fid, '%g', 3);
    fgetl(fid);
end

fclose(fid);
