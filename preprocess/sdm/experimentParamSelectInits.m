function experimentParamSelectInits(parsSdm, parsParametric, pars2d, pars3d)
    datasets   = {'aflw', 'lfpw'};
    algorithms = {'sdm', 'parametric', '2d', '3d'};
    
    function [results] = evalModel(iA, dataset, models, numInits)
        % Test for the current number of cascade steps
        [~, shapes, poses, errors, runtime] = main(iA, dataset, ...
            'performTesting', 1, ...
            'models', models ...
        );
        
        results = struct( ...
            'dataset',      dataset, ...
            'algorithm',    algorithms(iA), ...
            'numInits',     numInits, ...
            'models',       models, ...
            'runtime',      runtime, ...
            'shapes',       shapes, ...
            'poses',        poses, ...
            'errors',       errors ...
        );
    end
    
    for iD = [1 2]
        dataset = datasets{iD};
        
        for iI = [1 3 5 7 9 11]
            sfile = ['results/initsel_sdm_' dataset '_' num2str(iI) '.mat'];
            if exist(sfile) == 0
                % Genetare SDM initializations
                tmodels = load(['results/parsel_sdm_' dataset '_' num2str(parsSdm(iD,1)) '_' num2str(parsSdm(iD,2)) '.mat']);
                tmodels = tmodels.results.models;
                for i = 1:length(tmodels)
                    meanOffset = mean(tmodels(i).initShapes(:,:,1), 1);
                    centeredShape = reshape(tmodels(i).pca.mean, 2, [])';
                    tmodels(i).initShapes = repmat(centeredShape, [1 1 iI]);
                    for j = 1:iI
                        if iI == 1, rangle = 0;
                        else rangle = pi*(j-1)/(iI-1) - pi/2;
                        end
                        c = cos(rangle); s = sin(rangle);
                        tmodels(i).initShapes(:,:,j) = bsxfun(@plus, tmodels(i).initShapes(:,:,j) * [c s ; -s c], meanOffset);
                    end
                end

                % Run test
                results = evalModel(1, dataset, tmodels, iI);
                save(sfile, 'results');
                clear tmodels results;
            end

            sfile = ['results/initsel_parametric_' dataset '_' num2str(iI) '.mat'];
            if exist(sfile) == 0
                % Generate parametric initializations
                tmodels = load(['results/parsel_parametric_' dataset '_' num2str(parsParametric(iD,1)) '_' num2str(parsParametric(iD,2)) '.mat']);
                tmodels = tmodels.results.models;
                for i = 1:length(tmodels)
                    tmodels(i).initPoses = repmat(tmodels(i).initPoses(1,:), [iI 1]);
                    if iI == 1, tmodels(i).initPoses(:,tmodels(i).targets.idxsRotat(1)) = 0;
                    else tmodels(i).initPoses(:,tmodels(i).targets.idxsRotat(1)) = pi*(0:(iI-1))/(iI-1) - pi/2;
                    end
                end

                % Run test
                results = evalModel(2, dataset, tmodels, iI);
                save(sfile, 'results');
                clear tmodels results;
            end

            sfile = ['results/initsel_2d_' dataset '_' num2str(iI) '.mat'];
            if exist(sfile) == 0
                % Generate 2D initializations
                tmodels = load(['results/parsel_2d_' dataset '_' num2str(pars2d(iD,1)) '_' num2str(pars2d(iD,2)) '.mat']);
                tmodels = tmodels.results.models;
                for i = 1:length(tmodels)
                    tmodels(i).initPoses = repmat(tmodels(i).initPoses(1,:), [iI 1]);
                    if iI == 1, tmodels(i).initPoses(:,tmodels(i).targets.idxsRotat(1)) = 0;
                    else tmodels(i).initPoses(:,tmodels(i).targets.idxsRotat(1)) = pi*(0:(iI-1))/(iI-1) - pi/2;
                    end
                end

                % Run test
                results = evalModel(3, dataset, tmodels, iI);
                save(sfile, 'results');
                clear tmodels results;
            end

            sfile = ['results/initsel_3d_' dataset '_' num2str(iI) '.mat'];
            if exist(sfile) == 0
                % Generate 3D initializations
                tmodels = load(['results/parsel_3d_' dataset '_' num2str(pars3d(iD,1)) '_' num2str(pars3d(iD,2)) '.mat']);
                tmodels = tmodels.results.models;
                for i = 1:length(tmodels)
                    tmodels(i).initPoses = repmat(tmodels(i).initPoses(1,:), [iI 1]);
                    if iI == 1, tmodels(i).initPoses(:,tmodels(i).targets.idxsRotat(1)) = 0;
                    else tmodels(i).initPoses(:,tmodels(i).targets.idxsRotat(1)) = pi*(0:(iI-1))/(iI-1) - pi/2;
                    end
                end

                % Run test
                results = evalModel(4, dataset, tmodels, iI);
                save(sfile, 'results');
                clear tmodels results;
            end
        end
    end
end