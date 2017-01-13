function [model] = algorithmSDMTrain_original(trainData, testData, varargin)
    addpath(genpath('libs'))
    
    [nsL,nsD] = size(trainData(1).landmarks2d);
    sTrain  = length(trainData);
    sTest   = length(testData);
    
    % SET ALGORITHM PARAMETERS
    % ------------------------------------------------------------
    
    parsInit = {5 2};
    [T,aug] = deal(parsInit{:});
    
    nPars = length(varargin);
    for i = 1:2:nPars
        pN = varargin{i};
        pV = varargin{i+1};
        
        % Number of cascade steps
        if strcmpi(pN, 'numCascadeSteps')
            T = pV;
            
        % Level of training data boosing
        elseif strcmpi(pN, 'trainingDataBoosts')
            aug = pV;
            
        end
    end
    
    % ALIGN TRAINING INSTANCES
    % ------------------------------------------------------------
    
    % Get target geometries for train and test data
    trainShapeTargets = single(cat(3, trainData.landmarks2d));
    testShapeTargets = single(cat(3, testData.landmarks2d));
    
    % Find closest alignment shape and transforms to align train data
    [M,A,D] = generalizedProcrustes2D(trainShapeTargets);
    
    % Apply transforms to get aligned landmark information
    trainShapesAligned = zeros(sTrain, nsL*nsD);
    for i = 1:length(trainData)
        pv = trainData(i).landmarks2d * A{i}' + (diag(D{i})*ones(nsD,nsL))';
        trainShapesAligned(i,:) = reshape(pv', [], 1);
    end
    
    % ANALYSE SHAPE MODES OF VARIATION
    % ------------------------------------------------------------
    
    % Apply PCA to obtain the shape main modes of variation
    pcaMean = reshape(M, 1, []);
    [pcaTransform, pcaEigenvalues, pcaMean, trainShapeModes] = transformPCA(trainShapesAligned, 'variance', 0.95, 'mean', pcaMean);
    nShapeModes = size(pcaTransform,1);
    nParams = nShapeModes + 2 + 1 + 2;  % nShapeModes + location + rotation + scale
    
    % PREPARE PARAMETER INDEXS
    % ------------------------------------------------------------
    
    idxsShape = 1:nShapeModes;
    idxsLocat = [nShapeModes+1 nShapeModes+2];
    idxsRotat = nShapeModes+3;
    idxsScale = [nShapeModes+4 nShapeModes+5];

    % HELPER METHODS
    % ------------------------------------------------------------
    
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

    function [pose] = poseFromShape(shapeParams, dis, rot)
        pose = zeros(nParams,1);
        pose(idxsShape) = shapeParams;
        pose(idxsRotat) = atan2(rot(2,1), rot(2,2));
        pose(idxsScale) = [sqrt(rot(1,1)^2 + rot(1,2)^2), sqrt(rot(2,1)^2 + rot(2,2)^2)];
        pose(idxsLocat) = [
            (dis(2)*rot(1,2) - dis(1)*rot(2,2)) / (rot(2,1)*rot(1,2) - rot(1,1)*rot(2,2)) ...
            (dis(2)*rot(1,1) - dis(1)*rot(2,1)) / (rot(1,1)*rot(2,2) - rot(2,1)*rot(1,2)) ...
        ];
    end

    % FEATURES MEASUREMENT METHODS
    % ------------------------------------------------------------
    
    function [values] = features2D(image, shape)
        values = sd_sift(image, double(shape));
        values = values(:);
    end

    % FERN REGRESSOR METHODS
    % ------------------------------------------------------------
    
    function [regressor, results, mae] = regressorSDMTrain(fValues, shapeOrig, shapeTarg)
        nI = size(shapeTarg, 3);
        nT = nsL*nsD;

        % Initialize final regressor
        regressor = regressorSDMStub();
        
        % Get shape difference
        shapeDiff = shapeTarg - shapeOrig;
        shapeDiffVector = zeros(nI, nT);
        for iI = 1:nI
            shapeDiffVector(iI,:) = reshape(shapeDiff(:,:,iI)', 1, []);
        end
        
        % Reduce features dimensionality
        [pcaTfm, ~, pcaMn, fValuesPCA] = transformPCA(fValues, 'variance', 0.98);
        
        % Train SDM
        % ----------------------------------------
        
        meanValues = mean(fValuesPCA,1);
        meanShapes = mean(shapeDiffVector,1);
        
        weights = fValuesPCA\shapeDiffVector;
        offsets = meanShapes - meanValues * weights;
        
        % Build regressor
        % ----------------------------------------
        
        regressor.weights = weights;
        regressor.offsets = offsets;
        regressor.pcaMean = pcaMn;
        regressor.pcaTransform = pcaTfm;
        
        % Calculate output shapes
        tresults = bsxfun(@plus, fValuesPCA * regressor.weights, regressor.offsets);
        
        % Reorder output shapes
        results = zeros(nsL,nsD,nI);
        for iI = 1:nI
            results(:,:,iI) = reshape(tresults(iI,:)', nsD, nsL)';
        end
        
        % Evaluate overall accuracy
        mae = sqrt(sum(sum((shapeDiff - results) .^ 2, 1), 2));
    end

    function [results] = regressorSDMRegress(regressor, images, shapes)
        nElems = size(shapes,3);

        % Calculate features for all test data
        rfeatures = zeros(nElems, nsL*128);
        for iElem = 1:nElems
            rfeatures(iElem,:) = features2D(images{iElem}, shapes(:,:,iElem));
        end

        % Apply regressor
        rfeatures = bsxfun(@minus, rfeatures, regressor.pcaMean) * regressor.pcaTransform';
        tresults = bsxfun(@plus, rfeatures * regressor.weights, regressor.offsets);
        
        % Reshape results
        results = zeros(nsL,nsD,nElems);
        for iI = 1:nElems
            results(:,:,iI) = reshape(tresults(iI,:)', nsD, nsL)';
        end
    end

    function [model] = regressorSDMStub()
        model = struct( ...
            'offsets', [], ...
            'weights', [], ...
            'pcaMean', [], ...
            'pcaTransform', [] ...
        );
    end
    
    % PREPARE POSE PARAMETERS FOR EACH INSTANCE (AUGMENT DATA)
    % ------------------------------------------------------------
    nAug = aug*sTrain;
    dMap = 1 + floor((0:nAug-1) / aug);
    
    % Prepare target training shapes
    trainShapeTargets = trainShapeTargets(:,:,dMap);
    
    % Prepare target training poses
    trainPoseTargets = zeros(nAug,nParams);
    for i = 1:sTrain
        trainPoseTargets(i,:) = poseFromShape(trainShapeModes(i,:), D{i}, A{i});
    end
    trainPoseTargets = trainPoseTargets(dMap,:);
    
    % INITIALIZE MODEL STRUCTURE & SHAPE/POSE PARAMETERS FOLLOW-UP
    % ------------------------------------------------------------
    
    % Get initial shape/pose values
    basePoses = repmat(mean(trainPoseTargets,1), [10 1]);
    basePoses(:,idxsRotat) = pi * (0:9)/9 - pi/2;
    baseShapes = zeros(nsL,nsD,10);
    for iAug = 1:10
        baseShapes(:,:,iAug) = shapeFromPose(basePoses(iAug,:));
    end
    
    % Initialize model structure
    model = struct( ...
        'pca', struct( ....
            'mean',        pcaMean, ...
            'eigenvalues', pcaEigenvalues, ...
            'transform',   pcaTransform ...
        ), ...
        'targets', struct( ...
            'idxsShape', idxsShape, ...
            'idxsLocat', idxsLocat, ...
            'idxsRotat', idxsRotat, ...
            'idxsScale', idxsScale  ...
        ), ...
        'cascade', repmat(regressorSDMStub(), T, 1), ...
        'initShapes', baseShapes ...
    );

    % Create error vectors
    trainErrors = zeros(T+1,nAug);
    testErrors = zeros(T+1,sTest);
    
    % Get initial shape values for training/testing
    meanPose  = mean(trainPoseTargets, 1);
    testShape  = repmat(shapeFromPose(meanPose), [1 1 sTest]);
    trainShape = zeros(nsL, 2, nAug);
    for i = 1:nAug
        tpose = meanPose;
        tpose(:, idxsScale) = (0.75 + 0.5*rand(1,1)) * meanPose(idxsScale);
        tpose(:, idxsLocat) = (0.75 + 0.5*rand(1,2)) .* meanPose(idxsLocat);
        tpose(:, idxsRotat) = (rand - 0.5) * pi/2;
        trainShape(:,:,i) = shapeFromPose(tpose);
    end
    
    % Initialize vector holding the feature values at each cascade step
    feat = zeros(nAug, nsL*128, 'single');
    
    % TRAIN MODEL REGRESSORS
    % ------------------------------------------------------------
    
    % Show original error
    trainErrors(1,:) = sqrt(sum(sum((trainShape-trainShapeTargets) .^ 2, 1), 2));
    testErrors(1,:) = sqrt(sum(sum((testShape-testShapeTargets) .^ 2, 1), 2));
    disp(['Initial training reconstruction error: ' num2str(mean(trainErrors(1,:)))]);
    disp(['Initial testing reconstruction error:  ' num2str(mean(testErrors(1,:)))]);
    
    T
    for i = 1:T
        % Calculate features for all training images
        for j = 1:nAug
            feat(j,:) = features2D(trainData(dMap(j)).face, trainShape(:,:,j));
        end
        
        % Train regressor
        [regressor, out, mae] = regressorSDMTrain(feat, trainShape, trainShapeTargets);
        model.cascade(i) = regressor;
        
        % Update train/test data with the new regressor
        trainShape = trainShape + out;
        testShape = testShape + regressorSDMRegress(regressor, {testData.face}, testShape);

        % Show errors
        trainErrors(i+1,:) = sqrt(sum(sum((trainShape-trainShapeTargets) .^ 2, 1), 2));
        testErrors(i+1,:) = sqrt(sum(sum((testShape-testShapeTargets) .^ 2, 1), 2));
        disp(['Training reconstruction error at T=' num2str(i) ': ' num2str(mean(mae))]);
        disp(['Testing reconstruction error at T=' num2str(i) ':  ' num2str(mean(testErrors(i+1,:)))]);
    end
    
    return;
end