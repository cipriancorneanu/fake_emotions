function [ m, e ] = train( data, nObsVertices, nCascades, nTrainingBoosts )

    % Prepare folds
    N = length(data);
    nFolds = 5;
    idxs = splitData (N, nFolds);
    
    % Compute asm
    %ref_face = data(idxs); % Take first face as reference
    asm = compute_asm(data, idxs);
    
    % Compute norm
    norm = compute_norm(data);
    
    % For each fold
    for k = 1:nFolds
        fprintf('Training fold %d \n', k);
        train = data(idxs{k}.train);
        test = data(idxs{k}.test);
        
        % Train 
        obsVertices = 1:nObsVertices;

        [mod, err_fold] = algorithmSDMTrain_pose3d(train, test, ...
                    asm{k}, obsVertices, nCascades, norm , ...
                    'numCascadeSteps', nCascades, 'trainingDataBoosts', nTrainingBoosts);

        m{k} = mod;
        e{k} = err_fold;       
    end
end




