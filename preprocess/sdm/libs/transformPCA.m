function [W, Ev, M, pc] = transformPCA(data, varargin)
    parsInit = {1, 0, []};
    [variance,neigen,M] = deal(parsInit{:});
    
    nPars = length(varargin);
    for i = 1:2:nPars
        pN = varargin{i};
        pV = varargin{i+1};
        
        if strcmpi(pN, 'variance')
            variance = pV;
        elseif strcmpi(pN, 'numBases')
            neigen = pV;
        elseif strcmpi(pN, 'mean')
            M = pV;
        end
    end
    
    % Remove the mean variable-wise (row-wise)
    if isempty(M), M = mean(data, 1); end
    data = bsxfun(@minus, data, M);

    % Calculate eigenvectors W, and eigenvalues Ev
    [W, EvalueMatrix] = eig(cov(data));
    [Ev,iEv] = sort(diag(EvalueMatrix), 'descend');
    
    if neigen == 0
        % Keep only the desired variance
        Evc = cumsum(Ev / sum(Ev));
        neigen = min(sum(Evc < variance)+1, length(Ev));
    end
    
    W = W(:,iEv(1:neigen))';
    Ev = Ev(1:neigen);
    pc = data * W';
end