function [] = experimentParamSelect()
    datasets = {'aflw', 'lfpw'};
    algorithms = {'sdm', 'parametric', '2d', '3d'};
    numCascadeSteps = [7 6 5 4 3 2 1];
    
    for iD = [1 2]
        dataset = datasets{iD};
        
        for iA = [1 2 3 4]
            algorithm = algorithms{iA};
            
            for nBoosts = [1 5 10 15 20 25]
                disp(' ');
                disp('::: ////////////////////////////////////////////////////');
                disp(['::: DATA BOOSTS: ' num2str(nBoosts)]);
                disp(['::: ALGORITHM: ' algorithm]);
                disp(['::: DATASET: ' dataset]);
                disp('::: ////////////////////////////////////////////////////');
                disp(' ');
                
                % Load temporary models if exist
                if exist('temp_models.mat') == 2
                    models = load('temp_models.mat');
                    models = models.models;
                    
                % Create models if don't exist
                else
                    [models, ~, ~, ~, ~] = main(iA, dataset, ...
                        'numFolds', 10, ...
                        'numCascadeSteps', max(numCascadeSteps), ...
                        'numBoosts', nBoosts, ...
                        'performTesting', 0 ...
                    );
                
                    save('temp_models.mat', 'models');
                end

                for nCascadeSteps = numCascadeSteps
                    fname = ['results/parsel_' algorithm '_' dataset '_' num2str(nCascadeSteps) '_' num2str(nBoosts) '.mat'];
                    if exist(fname) ~= 0
                        continue;
                    end
                
                    % Keep cascade steps of interest only
                    tmodels = models;
                    for iM = 1:length(tmodels)
                        tmodels(iM).cascade = tmodels(iM).cascade(1:nCascadeSteps);
                    end
                    
                    % Test for the current number of cascade steps
                    [~, shapes, poses, errors, runtime] = main(iA, dataset, ...
                        'numCascadeSteps', max(numCascadeSteps), ...
                        'numBoosts', nBoosts, ...
                        'performTesting', 1, ...
                        'models', tmodels ...
                    );
                    
                    % Display error
                    disp(' ');
                    disp(['Overall mean error: ' num2str(mean(errors)) ' +- ' num2str(1.96 * std(errors) / sqrt(length(errors)))]);
                    
                    % Save results
                    results = struct( ...
                        'dataset',          dataset, ...
                        'algorithm',        iA, ...
                        'numCascadeSteps',  nCascadeSteps, ...
                        'numBoosts',        nBoosts, ...
                        'models',           tmodels, ...
                        'runtime',          runtime, ...
                        'shapes',           shapes, ...
                        'poses',            poses, ...
                        'errors',           errors ...
                    );

                    save(fname, 'results');
                end
                
                % Delete temporary models
                delete('temp_models.mat');
            end
        end
    end
end