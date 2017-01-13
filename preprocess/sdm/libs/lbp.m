function hist = lbp(img, lm, wSize)

R = [3 5 7 9 11];

C = wSize/2 + 1;
L = wSize;

img = uint32(img);

%row_max = size(region,1)-R+1;
%col_max = size(region,2)-R+1;
labels = zeros(1,length(lm));

% For every landmark
for i = 1:length(lm)
        
    % Crop window
    ctr = round(lm(i,:));
    A = img(ctr(1)-wSize/2:ctr(1)+wSize/2-1, ctr(2)-wSize/2:ctr(2)+wSize/2-1);
    
    labels(i) = [];
    
    % For every radius
    for r = R        
        A = A+1-A(C,C);

        A(A>0) = 1;

        % Squared neighbourhood, P = 8 samples. No uniform patterns. 
        labels(i) = [labels(i), A(C,r) + A(r,L)*2 + A(r,C)*4 + A(r,1)*8 + A(r,1)*16 + A(1,1)*32 + A(1,r)*64 + A(1,r)*128];                  
    end

end