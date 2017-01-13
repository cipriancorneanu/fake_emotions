function [M, A, D] = generalizedProcrustes2D(data)
    data(:,end+1,:) = 1;
    [L,d,n] = size(data);
    
    % Transform data format, L x d x n => dn x L
    Dm = reshape(data, L, d*n)';
    
    % Initialize transform matrices
    A = cell(n,1);
    A(:) = {eye(d)};
    
    % Optimize transforms and mean shape
    for nIter = 1:10
        M = (vertcat(A{:})' * vertcat(A{:})) \ (vertcat(A{:})' * Dm);
        for ir = 1:n
            A{ir} = data(:,:,ir)' * pinv(M);
        end
    end
    
    % Move mean to the coordinate origin
    dsp = mean(M(1:(d-1),:), 2);
    M = bsxfun(@minus, M(1:(d-1),:), dsp);
    
    % Split into transform + displacement
    D = cell(n,1);
    for ir = 1:n
        A{ir} = pinv(A{ir});
        D{ir} = A{ir}(1:(d-1),d) - dsp;
        A{ir} = A{ir}(1:(d-1),1:(d-1));
    end
end