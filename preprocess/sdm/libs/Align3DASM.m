function [fModel, fWeights, fValues] = Align3DASM(ASM3D, Data2D, varargin)
    nBases = length(ASM3D.eigenvalues);
    w = zeros(nBases, 1);
    Data2D(:,end+1) = 1;
    
    % Set algorithm parameters
    parsInit = {[1 2]};
    [iNor] = deal(parsInit{:});
    nPars = length(varargin);
    for i = 1:2:nPars
        pN = varargin{i};
        pV = varargin{i+1};
        
        % Normalization landmarks indexs
        if strcmpi(pN, 'normIndexs')
            iNor = pV;
        end
    end
    
    % Prepare transformed PCA mean and bases
    meanPCA = reshape(ASM3D.mean, 3, []);
    basesPCA = zeros([size(meanPCA) nBases]);
    for i = 1:nBases
        basesPCA(:,:,i) = reshape(ASM3D.transform(i,:), 3, []);
    end
    
    % Prepare error tracking variables
    mae = [];
    normFac = sqrt(sum((Data2D(iNor(1),:) - Data2D(iNor(2),:)) .^ 2));
    
    for it = 1:50
        % Generate current 3D shape
        Data3D = reshape(ASM3D.mean(:) + ASM3D.transform' * w, 3, [])';
        [amodel, newV] = align3D(Data2D(:,1:2)', Data3D');
        
        % Get alignment/rotation+scaling projection transforms
        tfm = amodel.projection(1:2,1:3);
        off = amodel.projection(1:2,4);
        rsc = tfm(1:2,1:3);
        
        % Calculate current error and stop if converged
        mae(it) = mean(sqrt(sum((newV(1:2,:)' - Data2D(:,1:2)) .^ 2, 2))) / normFac;
        if it>1 && mae(it) >= mae(it-1)
            break;
        end
        
        % Save current model
        fModel   = amodel;
        fWeights = w;
        fValues  = newV;
        
        % Project pca mean and bases
        prjMean = bsxfun(@plus, tfm * meanPCA, off);
        prjMean = reshape(prjMean(1:2,:), [], 1);
        prjBases = zeros(length(prjMean), nBases);
        for i = 1:nBases
            %tmp = bsxfun(@plus, tfm * basesPCA(:,:,i), off);
            tmp = rsc * basesPCA(:,:,i);
            prjBases(:,i) = reshape(tmp, [], 1);
        end
        
        % Optimize shape coefficients
        targ = reshape(Data2D(:,1:2)', [], 1) - prjMean;
        w = rescalePCAOutliers((prjBases \ targ)', ASM3D.eigenvalues)';
    end
    
    %h = plot(mae);
    %waitfor(h);
end