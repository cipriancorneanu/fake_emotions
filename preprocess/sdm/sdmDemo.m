function [models, finalShapes, finalPoses, errors, runtime] = sdmDemo(algorithm, dataset, varargin)
    % algorithm: Determines the type of algorithm to run.
    %       0 => Run CPR algorithm
    %       1 => Run SDM algorithm
    %       2 => Run SDM algorithm (pose regression)
    %       3 => Run SDM algorithm (pose regression, 3D fitting at the end)
    %       4 => Run SDM algorithm (3D pose regression)
    % 
    % In order to run aflw/lfpw, the optional parameter 'normIndexs' must
    % be passed specifying the indexs of the landmarks used for
    % normalization. These are:
    %       AFLW => [8 11]
    %       LFPW => [17 18]

    % Add 'libs' folder and subfolders to path
    addpath(genpath('libs'))
    
    %% --------------------------------------------------------------------
    %  -- Set algorithm parameters
    %  --------------------------------------------------------------------
    
    parsInit = {2, 5, 10, 1, [], 1};
    [nFolds,nCascadeSteps,nBoosts,doTesting,models,sO] = deal(parsInit{:});
    
    nPars = length(varargin);
    for i = 1:2:nPars
        pN = varargin{i};
        pV = varargin{i+1};
        
        if strcmpi(pN, 'numFolds')
            nFolds        = max(pV, 2);
        elseif strcmpi(pN, 'numCascadeSteps')
            nCascadeSteps = pV;
        elseif strcmpi(pN, 'numBoosts')
            nBoosts       = pV;
        elseif strcmpi(pN, 'performTesting')
            doTesting     = pV;
        elseif strcmpi(pN, 'models')
            models        = pV;
            nFolds        = length(models);
        elseif strcmpi(pN, 'dataset')
            data          = pV;
        elseif strcmpi(pN, 'showOutputs')
            sO            = pV;
        end
    end
    
    %% --------------------------------------------------------------------
    %  -- Read dataset-specific data
    %  --------------------------------------------------------------------
    
    if strcmpi(dataset, 'aflw')
        fpca  = 'data/aflw3DASM.mat';
        fdata = 'data/aflw.mat';
        normIndexs = [8 11];
    elseif strcmpi(dataset, 'lfpw')
        %fpca  = 'data/lfpw3DASM.mat';
        fdata = 'data/lfpw_relabeled.mat';
        normIndexs = [17 18];
    end
    
    % Load dataset
    if exist('data', 'var') == 0
        data = load(fdata);
        data = data.dataset;
    end
    
    % Load 3D ASM
    %facePCA = load(fpca);
    %facePCA = facePCA.facePCA;
    
    %% --------------------------------------------------------------------
    %  -- Algorithm execution method
    %  --------------------------------------------------------------------
    
    function [model, finalShapes, finalPoses, errors, runtime] = runAlgorithm(train, test, model)
        finalShapes = [];
        finalPoses  = [];
        errors      = [];
        runtime     = 0;
        
        % Select algorithm to execute
        switch algorithm
            case 0 % CPR algorithm
                % Train model
                if nargin < 3
                    if sO == 1, disp([char(10) 'Training CPR model:']); end
                    model = algorithmCPRTrain(train, test, ...
                        'numCascadeSteps',          100,    ... % 10
                        'numRegressors',            50,     ... % 500
                        'numPoolFeatures',          400,    ... % 400
                        'numRegressorFeatures',     5,      ... % 5
                        'numThreshCombinations',    1000,   ... % 1000
                        'trainingDataBoosts',       20      ... % 20
                    );

                     % Save current model
                    save('modelCPR.mat', 'model');
                end

                % Test model
                if doTesting == 1
                    disp([char(10) 'Testing CPR model:']);
                    algorihmCPRTest(model, {test.face}, cat(3, test.landmarks2d));
                end

            case 1 % SDM algorithm
                % Train model
                if nargin < 3
                    if sO == 1, disp([char(10) 'Training SDM model:']); end
                    model = algorithmSDMTrain_original(train, test, ...
                        'numCascadeSteps',          nCascadeSteps,  ... % 5
                        'trainingDataBoosts',       nBoosts         ... % 20
                    );

                    % Save current model
                    save('modelSDM.mat', 'model');
                end

                % Test model
                if doTesting == 1
                    if sO == 1, disp([char(10) 'Testing SDM model:']); end
                    [finalShapes, errors, runtime] = algorithmSDMTest_original(model, {test.face}, ...
                        'targets', cat(3, test.landmarks2d), ...
                        'normIndexs', normIndexs, ...
                        'showOutputs', 0 ...
                    );
                end

            case 2 % SDM algorithm (pose regression)
                % Train model
                if nargin < 3
                    if sO == 1, disp([char(10) 'Training SDM model (pose regression):']); end
                    model = algorithmSDMTrain_pose(train, test,    ...
                        'numCascadeSteps',          nCascadeSteps, ... % 5
                        'trainingDataBoosts',       nBoosts        ... % 10
                    );

                    % Save current model
                    save('modelSDMPose.mat', 'model');
                end
                
                % Test model
                if doTesting == 1
                    if sO == 1, disp([char(10) 'Testing SDM model (pose regression):']); end
                    [finalShapes, errors, runtime] = algorithmSDMTest_pose(model, {test.face}, ...
                        'targets', cat(3, test.landmarks2d), ...
                        'normIndexs', normIndexs, ...
                        'showOutputs', 0 ...
                    );
                end

            case 3 % SDM algorithm (pose regression, 3D fitting at the end)
                % Train model
                if nargin < 3
                    if sO == 1, disp([char(10) 'Training SDM model (pose regression, 3D fitting at the end):']); end
                    model = algorithmSDMTrain_pose2d(train, test, facePCA, ...
                        'numCascadeSteps',          nCascadeSteps,         ... % 5
                        'trainingDataBoosts',       nBoosts                ... % 10
                    );

                    % Save current model
                    save('modelSDM2d.mat', 'model');
                end

                % Test model
                if doTesting == 1
                    if sO == 1, disp([char(10) 'Testing SDM model (pose regression, 3D fitting at the end):']); end
                    [finalShapes, finalPoses, errors, runtime] = algorithmSDMTest_pose2d(model, {test.face}, ...
                        'targets', cat(3, test.landmarks3d), ...
                        'normIndexs', normIndexs, ...
                        'showOutputs', 0 ...
                    );
                end

            case 4 % SDM algorithm (3D pose regression)
                % Train model
                if nargin < 3
                    if sO == 1, disp([char(10) 'Training SDM model (3D pose regression):']); end
                    model = algorithmSDMTrain_pose3d(train, test, facePCA, ...
                        'numCascadeSteps',          nCascadeSteps,         ... % 5
                        'trainingDataBoosts',       nBoosts                ... % 10
                    );

                    % Save current model
                    save('modelSDM3d.mat', 'model');
                end

                % Test model
                if doTesting == 1
                    if sO == 1, disp([char(10) 'Testing SDM model (3D pose regression):']); end
                    [finalShapes, finalPoses, errors, runtime] = algorithmSDMTest_pose3d(model, {test.face}, ...
                        'targets', cat(3, test.landmarks3d), ...
                        'normIndexs', normIndexs, ...
                        'showOutputs', 0 ...
                    );
                end
        end
    end

    %% --------------------------------------------------------------------
    %  -- Execute algorithm
    %  --------------------------------------------------------------------
    
    finalShapes = [];
    finalPoses  = [];
    errors      = [];
    runtime     = 0;
    
    dSize = length(data);
    sSize = floor(dSize / nFolds);
    for i = 1:nFolds
        % Select train and test data
        train = data([1:((i-1)*sSize) (i*sSize+1):end]);
        test  = data(((i-1)*sSize+1) : (i*sSize));
        
        if sO == 1, disp(['EXECUTING FOLD ' num2str(i) '/' num2str(nFolds)]); end
        if length(models) >= i
            [tmodel, tshapes, tposes, terrors, truntime] = runAlgorithm(train, test, models(i));
        else
            [tmodel, tshapes, tposes, terrors, truntime] = runAlgorithm(train, test);
        end
        if sO == 1, disp(' '); end
        
        models      = cat(1, models, tmodel);
        finalShapes = cat(3, finalShapes, tshapes);
        finalPoses  = cat(1, finalPoses, tposes);
        errors      = cat(1, errors, terrors);
        runtime = runtime + truntime / dSize;
    end
end