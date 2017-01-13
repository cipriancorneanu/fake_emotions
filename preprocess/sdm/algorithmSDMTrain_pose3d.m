function [model, err] = algorithmSDMTrain_pose3d(trainData, testData, ASM3D, obsVert, nCascades, errNorm, varargin)
    
    nsL = size(trainData(1).landmarks,1);
    [nsO,nsD] = size(trainData(1).landmarks(obsVert,:));
    
    sTrain  = length(trainData);
    sTest   = length(testData);
    
    % SET ALGORITHM PARAMETERS
    % ------------------------------------------------------------
    
    parsInit = {5 1};
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
    
    % Get target geometries for train data
    trainShapeTargets = zeros(nsL, nsD, sTrain, 'single');
    for iI = 1:sTrain
        trainShapeTargets(:,:,iI) = trainData(iI).landmarks;
    end
    
    % Get target geometries for test data
    testShapeTargets = zeros(nsL, nsD, sTest, 'single');
    for iI = 1:sTest
        testShapeTargets(:,:,iI) = testData(iI).landmarks;
    end

    % ANALYSE SHAPE MODES OF VARIATION
    % ------------------------------------------------------------
    
    pcaTransform = ASM3D.transform;
    pcaEigenvalues = ASM3D.eigenvalues;
    pcaMean = ASM3D.mean;
    
    nShapeModes = size(pcaTransform,1);
    nParams = nShapeModes + 3 + 3 + 1; % nShapeModes + location + rotation + scale
    
    % GET MEAN SIFT WINDOW SIZE
    % ------------------------------------------------------------
    
    meanFace = reshape(pcaMean, nsD, nsL);
    mwSize = 32 * (mean(max(meanFace(1:2,:), [], 2)) / 90);
    
    % PREPARE PARAMETER INDEXS
    % ------------------------------------------------------------
    
    idxsShape = 1:nShapeModes;
    idxsLocat = (nShapeModes+1):(nShapeModes+2);
    idxsRotat = (nShapeModes+3):(nShapeModes+5);
    idxsScale = nShapeModes+6;

    % HELPER METHODS
    % ------------------------------------------------------------
    
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
        %S = reshape((pcaMean + poseParams(idxsShape) * pcaTransform), 3, 1024)';
        %S_ = bsxfun(@plus, reshape((pcaMean + poseParams(idxsShape) * pcaTransform), 3, 1024)', dsp);
        shape = bsxfun(@plus, reshape(pcaMean + poseParams(idxsShape) * pcaTransform, 3, [])' * tfm', dsp);
        %shape = reshape(pcaMean + poseParams(idxsShape) * pcaTransform, 3, [])'
    end

    function [pose] = poseFromShape(shapeParams, dis, rot, sca)
        pose = zeros(nParams,1);
        pose(idxsShape) = shapeParams;
        pose(idxsLocat) = dis;
        pose(idxsRotat) = rot;
        pose(idxsScale) = sca;
    end

    % FEATURES MEASUREMENT METHODS
    % ------------------------------------------------------------
    
    function [values] = features2D(image, shape2d, wSize)
        values = sd_sift(double(image), double(shape2d), 'winsize', wSize);
        
        %Labels = double(lbp(double(image), double(shape2d), 32));             
        %face_histograms = spatial_LBP_histogram ( Labels, Current_regions, his_len); % spatial LBP histogram      
        values = values(:);
    end

    % FERN REGRESSOR METHODS
    % ------------------------------------------------------------
    
    function [regressor, results, mae] = regressorSDMTrain(fValues, poseOrig, poseTarg)
        % Initialize final regressor
        regressor = regressorSDMStub();
        
        % Get pose difference
        poseDiff = poseTarg - poseOrig;
        
        % Reduce features dimensionality
        [pcaTfm, ~, pcaMn, fValuesPCA] = transformPCA(fValues, 'variance', 0.98);
        
        % Train SDM
        % ----------------------------------------
        
        meanValues = mean(fValuesPCA,1);
        meanPoses = mean(poseDiff,1);
        
        weights = fValuesPCA\poseDiff;
        offsets = meanPoses - meanValues * weights;
        
        % Build regressor
        % ----------------------------------------
        
        regressor.weights = weights;
        regressor.offsets = offsets;
        regressor.pcaMean = pcaMn;
        regressor.pcaTransform = pcaTfm;
        
        % Calculate output pose
        results = bsxfun(@plus, fValuesPCA * regressor.weights, regressor.offsets);
        
        % Evaluate overall accuracy
        mae = squeeze(sqrt(sum((poseDiff - results) .^ 2, 2)));
    end

    function [results] = regressorSDMRegress(regressor, images, shapes, poses)
        nElems = length(images);
        
        % Calculate features for all test data
        rfeatures = zeros(nElems, nsO*128);
        for iElem = 1:nElems
            wSize = round(mwSize * poses(iElem,idxsScale));
            rfeatures(iElem,:) = features2D(images{iElem}, shapes(obsVert,1:2,iElem), wSize);
        end

        % Apply regressor
        rfeatures = bsxfun(@minus, rfeatures, regressor.pcaMean) * regressor.pcaTransform';
        results = bsxfun(@plus, rfeatures * regressor.weights, regressor.offsets);
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
        idx = ASM3D.idxs.train(i);        
        trainPoseTargets(i,:) = poseFromShape( ...
            ASM3D.fValuesPCA(idx,:), ...
            ASM3D.pose(idx).offset(1:2), ...
            [ASM3D.pose(idx).rot.roll ASM3D.pose(idx).rot.pitch ASM3D.pose(idx).rot.yaw], ...
            ASM3D.pose(idx).scale ...
        );
    %{
        trainPoseTargets(i,:) = poseFromShape( ...
            trainData(i).pose3d.weights, ...
            trainData(i).pose3d.offsets(1:2), ...
            [trainData(i).pose3d.roll trainData(i).pose3d.pitch trainData(i).pose3d.yaw], ...
            trainData(i).pose3d.scale ...
        );
     %}
    end
    trainPoseTargets = trainPoseTargets(dMap,:);
    
    % Prepare target test poses
    testPoseTargets = zeros(numel(testData),nParams);
    for i = 1:sTest
        idx = ASM3D.idxs.test(i);
        trainPoseTargets(i,:) = poseFromShape( ...
            ASM3D.fValuesPCA(idx,:), ...
            ASM3D.pose(idx).offset(1:2), ...
            [ASM3D.pose(idx).rot.roll ASM3D.pose(idx).rot.pitch ASM3D.pose(idx).rot.yaw], ...
            ASM3D.pose(idx).scale ...
        );
    
        %{
        testPoseTargets(i,:) = poseFromShape( ...
            testData(i).pose3d.weights, ...
            testData(i).pose3d.offsets(1:2), ...
            [testData(i).pose3d.roll testData(i).pose3d.pitch testData(i).pose3d.yaw], ...
            testData(i).pose3d.scale ...
        );
        %}
    end
    
    % INITIALIZE MODEL STRUCTURE & SHAPE/POSE PARAMETERS FOLLOW-UP
    % ------------------------------------------------------------
    
    % Get initial shape/pose values
    basePoses = repmat(mean(trainPoseTargets,1), [10, 1]);
    basePoses(:,idxsRotat(1)) = pi * (0:9)/9 - pi/2;
    
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
            'idxsScale', idxsScale ...
        ), ...
        'cascade', repmat(regressorSDMStub(), T, 1), ...
        'initPoses', basePoses ...
    );
    

    % Create error vectors
    trainErrors = zeros(T+1,nAug);
    testErrors = zeros(T+1,sTest);

    % Get initial pose values for training/testing
    meanPose  = mean(trainPoseTargets, 1);
    testPose  = repmat(meanPose, [sTest 1]);
    trainPose = repmat(meanPose, [nAug 1]);
    trainPose(:, idxsScale) = bsxfun(@times, 0.75 + 0.5*rand(nAug,1), meanPose(idxsScale));
    trainPose(:, idxsLocat) = bsxfun(@times, 0.75 + 0.5*rand(nAug,2), meanPose(idxsLocat));
    trainPose(:, idxsRotat(1)) = (rand(nAug,1) - 0.5) * pi/2;
    
    % Get initial shape values for training/testing
    testShape  = repmat(shapeFromPose(meanPose), [1 1 sTest]);
    testShapeShape = zeros(nsL, 3, numel(testData));
    testRotationShape = zeros(nsL, 3, numel(testData));
    testLocationShape = zeros(nsL, 3, numel(testData));
    trainShape = zeros(nsL, 3, nAug);
    
    for i = 1:nAug
        trainShape(:,:,i) = shapeFromPose(trainPose(i,:));
    end
    
    % Initialize vector holding the feature values at each cascade step
    feat = zeros(nAug, nsO*128, 'single');
    
    % TRAIN MODEL REGRESSORS
    % ------------------------------------------------------------
    
    for i = 1:nCascades
        % Calculate features for all training images
        for j = 1:nAug
            wSize = round(mwSize * trainPose(j,idxsScale));
            feat(j,:) = features2D(trainData(dMap(j)).rgb, trainShape(obsVert,1:2,j), wSize);
        end
        
        % Train regressor
        [regressor, out, mae] = regressorSDMTrain(feat, trainPose, trainPoseTargets);
        model.cascade(i) = regressor;
        
        % Evaluate regressor over train data
        trainPose = trainPose + out;
        for j = 1:nAug
            trainShape(:,:,j) = shapeFromPose(trainPose(j,:));
        end
        
        % Evaluate regressor over test data
        testPose = testPose + regressorSDMRegress(regressor, {testData.rgb}, testShape, testPose);
        testPose(:,idxsShape) = rescalePCAOutliers(testPose(:,idxsShape),...
                                                    pcaEigenvalues);
        for j = 1:sTest
            testShape(:,:,j) = shapeFromPose(testPose(j,:));
            
            shapeTestPose = [testPose(:, idxsShape) testPoseTargets(:, [idxsLocat idxsRotat  idxsScale]) zeros(length(testPose),1)];          
            rotationTestPose = [ testPoseTargets(:, [idxsShape, idxsLocat])...
                                      testPose(:,idxsRotat) testPoseTargets(:,idxsScale) zeros(length(testPose),1)];
            locationTestPose = [ testPoseTargets(:, idxsShape)...
                                      testPose(:, idxsLocat) testPoseTargets(:, idxsRotat)...
                                      testPose(:, idxsScale) zeros(length(testPose),1)];
                                  
            testShapeShape(:,:,j) = shapeFromPose(shapeTestPose(j,:));                      
            testRotationShape(:,:,j) = shapeFromPose(rotationTestPose(j,:)); 
            testLocationShape(:,:,j) = shapeFromPose(locationTestPose(j,:)); 
        end

        % Visualize regression at every step
        %TODO
        img_idx = 1;
        visualizeShape = shapeFromPose(testPose(img_idx,:));
        
        
        %{
        figure(1); imshow(testData(img_idx).face); hold on;
                    scatter3(testShapeShape(:,1,img_idx), testShapeShape(:,2,img_idx),...
                        testShapeShape(:,3,img_idx)); hold on;
                    scatter3(testShapeTargets(:,1,img_idx), testShapeTargets(:,2,img_idx),...
                        testShapeTargets(:,3,img_idx));
        
        figure(2); imshow(testData(img_idx).face); hold on;
                    scatter3(testRotationShape(:,1,img_idx), testRotationShape(:,2,img_idx),...
                        testRotationShape(:,3,img_idx)); hold on;
                    scatter3(testShapeTargets(:,1,img_idx), testShapeTargets(:,2,img_idx),...
                        testShapeTargets(:,3,img_idx));
        
        figure(3); imshow(testData(img_idx).face); hold on;
                    scatter3(testLocationShape(:,1,img_idx), testLocationShape(:,2,img_idx),...
                        testLocationShape(:,3,img_idx)); hold on;
                    scatter3(testShapeTargets(:,1,img_idx), testShapeTargets(:,2,img_idx),...
                        testShapeTargets(:,3,img_idx));
        
        figure(); imshow(testData(img_idx).rgb); hold on;
                    %scatter3(visualizeShape(:,1), visualizeShape(:,2),...
                    %    visualizeShape(:,3)); hold on;
                    %scatter3(testShapeTargets(:,1,img_idx), testShapeTargets(:,2,img_idx),...
                    %    testShapeTargets(:,3,img_idx)); hold on;
                    drawFace(visualizeShape, 'r'); hold on;
                    drawFace(testShapeTargets(:,:,img_idx), 'g', 'PrintShape');
        %}            
        %% ERROR      
        err_train = sum(mean(mean(sqrt( (trainShape-trainShapeTargets) .^ 2), 3),2))/errNorm;
        err_test = sum(mean(mean(sqrt( (testShape-testShapeTargets) .^ 2), 3),2))/errNorm;

        err_test_shape = sum(mean(mean(sqrt( (testShapeShape-testShapeTargets) .^ 2), 3),2))/errNorm;
        err_test_rotation = sum(mean(mean(sqrt( (testRotationShape-testShapeTargets) .^ 2), 3),2))/errNorm;
        err_test_location = sum(mean(mean(sqrt( (testLocationShape-testShapeTargets) .^ 2), 3),2))/errNorm;
        
        % Show 
        %trainErrors(i+1,:) = mean(sqrt(sum((trainShape(obsVert,:,:)-trainShapeTargets) .^ 2, 2)), 1);
        %testErrors(i+1,:) = mean(sqrt(sum((testShape(obsVert,:,:)-testShapeTargets) .^ 2, 2)), 1);
        %{
        disp(['*************     T=' num2str(i) '      **************']);
        
        disp(['TRAINING ERROR:' num2str(err_train)]);
        split_err_train = mean(mean(sqrt( (trainShape-trainShapeTargets) .^ 2), 1),3);
        disp(['Training error in x = ' num2str(i) ': ' num2str(split_err_train(1))]);
        disp(['Training error in y = ' num2str(i) ': ' num2str(split_err_train(2))]);
        disp(['Training error in z = ' num2str(i) ': ' num2str(split_err_train(3))]);
        disp(['TESTING ERROR:' num2str(err_test)]);
        split_err_test = mean(mean(sqrt( (testShape-testShapeTargets) .^ 2), 1),3);
        disp(['Testing error in x = ' num2str(i) ': ' num2str(split_err_test(1))]);
        disp(['Testing error in y = ' num2str(i) ': ' num2str(split_err_test(2))]);
        disp(['T error in z = ' num2str(i) ': ' num2str(split_err_test(3))]);
        %}
        
        % Show error decomposition
        errShape = mean(sqrt(sum((testPose(:,idxsShape)./testPoseTargets(:,idxsShape) - 1 ) .^ 2, 2)));
        errRotat = mean(sqrt(sum((testPose(:,idxsRotat)-testPoseTargets(:,idxsRotat)) .^ 2, 2)));
        errLocat = mean(sqrt(sum((testPose(:,idxsLocat)-testPoseTargets(:,idxsLocat)) .^ 2, 2)));        
        errScale = mean(sqrt(sum((testPose(:,idxsScale)./testPoseTargets(:,idxsScale) - 1 ) .^ 2, 2)));
        %disp(['    Error decomposed: shape=' num2str(errShape) ' rotation=' num2str(errRotat) ' location=' num2str(errLocat)]);
        
        err{i}.whole = err_test; % MSE normated by area of the face
        err{i}.rotat = errRotat*180/pi; % Cumulative error (pitch, roll, yaw) in degrees
        err{i}.locat = errLocat; % Cumulative error (in x and y) in pixels
        err{i}.shape = errShape; % Error of shape parameters in % 
        err{i}.scale = errScale; % Error of scale parameter in %
        err{i}.wrotat = err_test_rotation; % MSE normater by area of the face for rotation estimation
        err{i}.wlocat = err_test_location; % MSE normater by area of the face for location estimation
        err{i}.wshape = err_test_shape; % MSE normater by area of the face for shape estimation
        
    end
end