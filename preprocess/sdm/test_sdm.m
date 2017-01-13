addpath(genpath('./bosphorus'));
addpath(genpath('./libs'));

% Load path
load_path = '/Users/cipriancorneanu/Documents/Bosphorus/MatlabDB/rotated';

% Load ASM model
load(strcat(load_path, filesep, 'asm3d.mat'));
load(strcat(load_path, filesep, 'mean_face_area.mat'));

numPersBatch = 1;

% Load batches
for batch = 1:20
    name_batch = strcat('rotated_batch', num2str((batch-1)*numPersBatch+1), '_',...
                                 num2str(batch*numPersBatch));
    batches{batch} = load(strcat(load_path, filesep, name_batch));
end

% Implement N-fold validation

N = 10;
L = length(batches);
nfold = round(L/N);
trainTestSplit = 0.9;

test = []; train = [];

for k = 2:2%N
    
    allIdx = 1:L;
    testIdx = (k-1)*nfold+1:k*nfold;
    trainIdx = [1:(k-1)*nfold  k*nfold+1 : allIdx(end)];

    for i = trainIdx
        train = [train batches{i}.out];
    end

    for i = testIdx
        test = [test batches{i}.out];
    end

    % Train
    slctStep = 16;
    obsVertices = 1:slctStep:length(train(1).landmarks3d);

    [model, err_fold] = algorithmSDMTrain_pose3d(train, test, ...
                asm, obsVertices, MEAN_FACE_AREA , 'numCascadeSteps', 5, 'trainingDataBoosts', 1);
            
    m{k} = model;
    e{k} = err_fold;
end
        

