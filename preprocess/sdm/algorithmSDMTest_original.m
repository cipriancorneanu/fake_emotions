function [finalShapes, errors, runtime] = algorithmSDMTest_original(model, images, varargin)
    nI = length(images);
    nsL = size(model.initShapes, 1);
    
    % SET ALGORITHM PARAMETERS
    % ------------------------------------------------------------
    
    parsInit = {length(model.cascade), size(model.initShapes, 3), [1 2], 1};
    [T,nP,iNor,sO] = deal(parsInit{:});
    
    nPars = length(varargin);
    for i = 1:2:nPars
        pN = varargin{i};
        pV = varargin{i+1};
        
        if strcmpi(pN, 'numCascadeSteps')
            T = min(pV, T);
        elseif strcmpi(pN, 'numInitialPoses')
            nP = pV;
        elseif strcmpi(pN, 'showOutputs')
            sO = pV;
        elseif strcmpi(pN, 'normIndexs')
            iNor = pV;
        elseif strcmpi(pN, 'targets')
            targetShapes = pV;
        end
    end

    % HELPER METHODS
    % ------------------------------------------------------------
    
    function [distances] = getDistancesMatrix(poses)
        numPoses = size(poses,3);
        
        distances = zeros(numPoses,numPoses);
        for ii = 1:numPoses
            for jj = (ii+1):numPoses
                distances(ii,jj) = sum(sqrt(sum((poses(:,:,ii) - poses(:,:,jj)) .^ 2, 2)));
                distances(jj,ii) = distances(ii,jj);
            end
        end
    end

    function [values] = features2D(image, shape)
        %values = xx_sift(image, double(shape));
        values = sd_sift(double(image), double(shape));%cip
        values = values(:);
    end

    function [results] = regressorSDMRegress(regressor, images, shapes)
        nElems = length(images);
            
        % Calculate features for all test data
        rfeatures = zeros(nElems, nsL*128);
        for iElem = 1:nElems
            rfeatures(iElem,:) = features2D(images{iElem}, shapes(:,:,iElem));
        end

        % Apply regressor
        rfeatures = bsxfun(@minus, rfeatures, regressor.pcaMean) * regressor.pcaTransform';
        tresults = bsxfun(@plus, rfeatures * regressor.weights, regressor.offsets);

        % Reshape results
        results = zeros(nsL, 2, nElems);
        for iElem = 1:nElems
            results(:,:,iElem) = reshape(tresults(iElem,:)', 2, nsL)';
        end
    end

    %% --------------------------------------------------------------------
    %  -- RUN ALGORITHM
    %  --------------------------------------------------------------------

    tic;

    % PREPARE INITIAL SHAPES
    % ------------------------------------------------------------

    assig = reshape(repmat(1:nI, [nP 1]), [], 1);
    shapes = repmat(model.initShapes(:,:,1:nP), [1 1 nI]);
    
    % APPLY REGRESSOR TO INITIAL POSES
    % ------------------------------------------------------------
    
    % Apply regessors to each element
    for iT = 1:T
        if sO == 1, disp(['Executing testing step ' num2str(iT) '...']); end
        delta_shape = regressorSDMRegress(model.cascade(iT), images(assig), shapes);
        shapes = shapes + regressorSDMRegress(model.cascade(iT), images(assig), shapes);
    end
    
    % Get centroid shapes
    finalShapes = zeros(nsL, 2, nI);
    for iI = 1:nI
        offs = (iI-1)*nP;

        % Get centroid pose from shape distances matrix
        [~,iP] = min(sum(getDistancesMatrix(shapes(:,:,offs+(1:nP))), 2));
        finalShapes(:,:,iI) = shapes(:,:,offs+iP);
    end
    
    if sO == 1
        plotResults(images, finalShapes);
    end
    
    runtime = toc;
    
    %% --------------------------------------------------------------------
    %  -- EVALUATE ERRORS IF GROUND TRUTH PRESENT
    %  --------------------------------------------------------------------
    
    % Evaluate errors
    errors = [];
    if exist('targetShapes', 'var')
        errors = squeeze(mean(sqrt(sum((finalShapes - targetShapes) .^ 2, 2)), 1) ./ sqrt(sum((targetShapes(iNor(1),:,:) - targetShapes(iNor(2),:,:)) .^ 2, 2)));
        merr = mean(errors);
        stdev = std(errors);
        if sO == 1, disp(['Testing reconstruction error: ' num2str(merr) ' +- ' num2str(1.96*stdev / sqrt(length(errors)))]); end
    end
end