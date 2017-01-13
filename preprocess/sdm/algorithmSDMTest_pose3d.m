function [finalShapes, finalPoses, errors, runtime] = algorithmSDMTest_pose3d(model, images, varargin)
    nI = length(images);
    nsL = numel(model.pca.mean) / 3;
    
    pcaMean = model.pca.mean;
    pcaEigenvalues = model.pca.eigenvalues;
    pcaTransform = model.pca.transform;
    
    % GET MEAN SIFT WINDOW SIZE
    % ------------------------------------------------------------
    
    meanFace = reshape(pcaMean, 3, nsL);
    mwSize = 32 * (mean(max(meanFace(1:2,:), [], 2)) / 90);
    
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
        dsp = [poseParams(idxsLocat) 0];
        rot = poseParams(idxsRotat);
        sca = poseParams(idxsScale);

        s3 = sin(rot(3));   c3 = cos(rot(3));
        s2 = sin(rot(2));   c2 = cos(rot(2));
        s1 = sin(rot(1));   c1 = cos(rot(1));
        
        tfm = sca * [ ...
            c3*c1+s3*s2*s1    c3*s1-s3*s2*c1    s3*c2   ; ...
            -c2*s1            c2*c1             s2      ; ...
            -s3*c1+c3*s2*s1   -s3*s1-c3*s2*c1   c3*c2   ; ...
        ];
        
        % Get shape from pose parameters
        shape = bsxfun(@plus, reshape(pcaMean + poseParams(idxsShape) * pcaTransform, 3, [])' * tfm', dsp);
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
            wSize = round(mwSize * poses(iElem,idxsScale));
            rfeatures(iElem,:) = features2D(images{iElem}, shapes(:,1:2,iElem), wSize);
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
    shapes = zeros(nsL,3,nP);
    for iS = 1:nP
        shapes(:,:,iS) = shapeFromPose(poses(iS,:));
    end
    shapes = repmat(shapes, [1 1 nI]);
    
    % APPLY REGRESSOR TO INITIAL POSES
    % ------------------------------------------------------------
    
    % Apply regessors to each element
    for iT = 1:T
        if sO == 1, disp(['Executing testing step ' num2str(iT) '...']); end
        poses = poses + regressorSDMRegress(model.cascade(iT), images(assig), shapes, poses);
        %poses(:,idxsShape) = rescalePCAOutliers(poses(:,idxsShape), pcaEigenvalues);
        
        % Get shapes for the image poses
        for iI = 1:(nI*nP)
            shapes(:,:,iI) = shapeFromPose(poses(iI,:));
        end
    end
    
    % Plot fitted 3d faces
    finalPoses = repmat(struct('roll', 0, 'pitch', 0, 'yaw', 0), nI, 1);
    finalShapes = zeros(nsL, 3, nI);
    for iI = 1:nI
        % Get centroid shape from shape distances matrix
        offs = (iI-1)*nP;
        [~,iP] = min(sum(getDistancesMatrix(shapes(:,:,offs+(1:nP))), 2));
        finalShapes(:,:,iI) = shapes(:,:,offs+iP);
        centroidPose  = poses(offs+iP,:);
        
        dsp = centroidPose(idxsLocat);
        rot = centroidPose(idxsRotat);
        sca = centroidPose(idxsScale);
        
        % Get face pose
        finalPoses(iI).roll  = rot(1);
        finalPoses(iI).pitch = rot(2);
        finalPoses(iI).yaw   = rot(3);
        
        if sO == 1
            roll    = [ cos(rot(1)) sin(rot(1)) 0       ; -sin(rot(1)) cos(rot(1)) 0      ; 0 0 1                      ];
            pitch   = [ 1 0 0                           ; 0 cos(rot(2)) sin(rot(2))       ; 0 -sin(rot(2)) cos(rot(2)) ];
            yaw     = [ cos(rot(3)) 0 sin(rot(3))       ; 0 1 0                           ; -sin(rot(3)) 0 cos(rot(3)) ];
            rmat = yaw * pitch * roll;

            h = plotFittedFace(images{iI}, finalShapes(:,:,iI), rmat);
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