function [M, A, D, out] = generalizedProcrustes3D(data, varargin)
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
        if nargin > 1
            %[~,~,tfm] = procrustes(varargin{1}(2:end,:)', M(1:d-1,varargin{1}(1,:))');
            %M = bsxfun(@plus, tfm.b * tfm.T' * M(1:d-1,:), tfm.c(1,:)');
            dAnch = size(varargin{1}, 1) - 1;
            [~,~,tfm] = procrustes(varargin{1}(2:end,:)', M(1:dAnch,varargin{1}(1,:))');
            M(1:dAnch,:) = bsxfun(@plus, tfm.b * tfm.T' * M(1:dAnch,:), tfm.c(1,:)');
        end
        for ir = 1:n
            %A{ir} = data(:,:,ir)' * pinv(M);
            [~,~,tfm] = procrustes(M(1:d-1,:)', data(:,1:d-1,ir));
            A{ir} = eye(d);
            A{ir}(1:d-1,1:d-1) = tfm.b * tfm.T';
            A{ir}(1:d-1,d) = tfm.c(1,:);
        end
    end
    
    % Move mean to the coordinate origin
    dsp = mean(M(1:(d-1),:), 2);
    M = bsxfun(@minus, M(1:(d-1),:), dsp);
    
    % If aligned data is requested
    if nargout > 3
        out = zeros(L,d-1,n);
        for ir = 1:n
            out(:,:,ir) = bsxfun(@minus, data(:,1:d,ir) * A{ir}(1:3,:)', dsp');
        end
    end
    
    % Split into transform + displacement
    D = cell(n,1);
    for ir = 1:n
        A{ir} = pinv(A{ir});
        D{ir} = A{ir}(1:(d-1),d) - dsp;
        A{ir} = A{ir}(1:(d-1),1:(d-1));
    end
end