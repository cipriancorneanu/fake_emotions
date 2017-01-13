function [finalShapes, finalPoses, errors, runtime] = algorithmSDMTest_pose2d(model, images, varargin)
    nI = length(images);
    nsL = numel(model.pca.mean) / 2;
    
    pcaMean = model.pca.mean;
    pcaEigenvalues = model.pca.eigenvalues;
    pcaTransform = model.pca.transform;
    
    % GET MEAN SIFT WINDOW SIZE
    % ------------------------------------------------------------
    
    meanFace = reshape(pcaMean, 2, nsL);
    mwSize = 32 * (mean(max(meanFace, [], 2)) / 90);
    
    % PREPARE PARAMETER INDEXS
    % ------------------------------------------------------------
    
    idxsShape = model.targets.idxsShape;
    idxsLocat = model.targets.idxsLocat;
    idxsRotat = model.targets.idxsRotat;
    idxsScale = model.targets.idxsScale;
    
    % SET ALGORITHM PARAMETERS
    % ------------------------------------------------------------
    
    parsInit = {length(model.cascade), size(model.initPoses, 1), [1 2], 1};
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

    function [shape] = shapeFromPose(poseParams)
        dsp = poseParams(idxsLocat);
        rot = poseParams(idxsRotat);
        sca = poseParams(idxsScale);
        
        tfm = [sca(1)*cos(rot) -sca(1)*sin(rot) 0 ; sca(2)*sin(rot) sca(2)*cos(rot) 0 ; 0 0 1];
        tfm(1:2,3) = [tfm(1,1)*dsp(1)+tfm(1,2)*dsp(2) tfm(2,1)*dsp(1)+tfm(2,2)*dsp(2)];
        tfm = pinv(tfm);
        
        % Get shape from pose parameters
        shape = bsxfun(@plus, reshape(pcaMean + poseParams(idxsShape) * pcaTransform, 2, [])' * tfm(1:2,1:2)', tfm(1:2,3)');
    end

    function [values] = features2D(image, shape, wSize)
        values = xx_sift(image, double(shape), 'winsize', wSize);
        values = values(:);
    end

    function [results] = regressorSDMRegress(regressor, images, shapes, poses)
        nElems = length(images);

        % Calculate features for all test data
        rfeatures = zeros(nElems, nsL*128);
        for iElem = 1:nElems
            wSize = round(mwSize * mean(poses(iElem,idxsScale)));
            rfeatures(iElem,:) = features2D(images{iElem}, shapes(:,:,iElem), wSize);
        end

        % Apply regressor
        rfeatures = bsxfun(@minus, rfeatures, regressor.pcaMean) * regressor.pcaTransform';
        results = bsxfun(@plus, rfeatures * regressor.weights, regressor.offsets);
    end

    %% --------------------------------------------------------------------
    %  -- RUN ALGORITHM
    %  --------------------------------------------------------------------

    tic;

    % PREPARE INITIAL POSES
    % ------------------------------------------------------------

    assig = reshape(repmat(1:nI, [nP 1]), [], 1);

    % Create initial poses
    poses = model.initPoses(1:nP,:);
    poses = repmat(poses, [nI 1]);
    
    % Create initial shapes
    shapes = zeros(nsL,2,nP);
    for iS = 1:nP
        shapes(:,:,iS) = shapeFromPose(poses(iS,:));
    end
    shapes = repmat(shapes, [1 1 nI]);
    
    % APPLY REGRESSOR TO INITIAL POSES
    % ------------------------------------------------------------
    
    % Apply regessors to each element
    for iT = 1:T
        disp(['Executing testing step ' num2str(iT) '...']);
        poses = poses + regressorSDMRegress(model.cascade(iT), images(assig), shapes, poses);
        poses(:,idxsShape) = rescalePCAOutliers(poses(:,idxsShape), pcaEigenvalues);
        
        % Get shapes from regressed poses
        for iI = 1:(nI*nP)
            shapes(:,:,iI) = shapeFromPose(poses(iI,:));
        end
    end
    
    % Find 3D-ASM projection to fitted 2D data
    finalPoses = repmat(struct('roll', 0, 'pitch', 0, 'yaw', 0), nI, 1);
    finalShapes = zeros(nsL, 3, nI);
    for iI = 1:nI
        % Get centroid shape from shape distances matrix
        offs = (iI-1)*nP;
        [~,iP] = min(sum(getDistancesMatrix(shapes(:,:,offs+(1:nP))), 2));
        centroidShape = shapes(:,:,offs+iP);
        
        % Fit 3D geometry
        [proj, w] = Align3DASM(model.ASM3D, centroidShape);
        
        % Get face pose
        finalPoses(iI).roll  = -atan2(proj.R(2,1), proj.R(1,1));
        finalPoses(iI).pitch = -atan2(proj.R(3,2), proj.R(3,3));
        finalPoses(iI).yaw   =  atan2(-proj.R(3,1), sqrt(proj.R(3,2)^2 + proj.R(3,3)^2));
        
        % Get face 3D geometry
        shape = reshape(model.ASM3D.mean + w' * model.ASM3D.transform, 3, []);
        shape(4,:) = 1;
        shape = (proj.transform * shape)';
        finalShapes(:,:,iI) = shape(:,1:3);
        
        if sO == 1
            h = plotFittedFace(images{iI}, centroidShape, model.ASM3D, proj, w);
            waitfor(h);
        end
    end
    
    runtime = toc;
    
    %% --------------------------------------------------------------------
    %  -- EVALUATE ERRORS IF GROUND TRUTH PRESENT
    %  --------------------------------------------------------------------
    
    % Evaluate error at current cascade step
    errors = [];
    if exist('targetShapes', 'var')
        errors = squeeze(mean(sqrt(sum((finalShapes - targetShapes) .^ 2, 2)), 1) ./ sqrt(sum((targetShapes(iNor(1),:,:) - targetShapes(iNor(2),:,:)) .^ 2, 2)));
        merr = mean(errors);
        stdev = std(errors);
        disp(['Testing reconstruction error: ' num2str(merr) ' +- ' num2str(1.96*stdev)]);
    end
end