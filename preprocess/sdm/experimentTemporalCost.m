function [results] = experimentTemporalCost()
    NRUNS = 32;
    
    results = struct( ...
        'aflw', struct( ....
            'sdm', zeros(1,NRUNS), ...
            'parametric', zeros(1,NRUNS), ...
            'd2d', zeros(1,NRUNS), ...
            'd3d', zeros(1,NRUNS) ...
        ), 'lfpw', struct( ...
            'sdm', zeros(1,NRUNS), ...
            'parametric', zeros(1,NRUNS), ...
            'd2d', zeros(1,NRUNS), ...
            'd3d', zeros(1,NRUNS) ...
        ) ...
    );

%     clear models;
%     models = load('results/initsel_sdm_aflw_1.mat');
%     for iI = 1:NRUNS
%         disp(['SDM AFLW RUN ' num2str(iI) '/' num2str(NRUNS)]);
%         [~,~,~,~,runtime] = main(1, 'aflw', 'performTesting', 1, 'models', models.results.models, 'showOutputs', 0);
%         results.aflw.sdm(iI) = runtime;
%     end
%     disp(['MEAN ERROR: ' num2str(mean(results.aflw.sdm))]);
% 
%     clear models;
%     models = load('results/initsel_sdm_lfpw_1.mat');
%     for iI = 1:NRUNS
%         disp(['SDM LFPW RUN ' num2str(iI) '/' num2str(NRUNS)]);
%         [~,~,~,~,runtime] = main(1, 'lfpw', 'performTesting', 1, 'models', models.results.models, 'showOutputs', 0);
%         results.lfpw.sdm(iI) = runtime;
%     end
%     disp(['MEAN ERROR: ' num2str(mean(results.lfpw.sdm))]);

    clear models;
    models = load('results/initsel_parametric_aflw_1.mat');
    for iI = 1:NRUNS
        disp(['PARAMETRIC AFLW RUN ' num2str(iI) '/' num2str(NRUNS)]);
        [~,~,~,~,runtime] = main(2, 'aflw', 'performTesting', 1, 'models', models.results.models, 'showOutputs', 0);
        results.aflw.parametric(iI) = runtime;
    end
    disp(['MEAN ERROR: ' num2str(mean(results.aflw.parametric))]);

    clear models;
    models = load('results/initsel_parametric_lfpw_1.mat');
    for iI = 1:NRUNS
        disp(['PARAMETRIC LFPW RUN ' num2str(iI) '/' num2str(NRUNS)]);
        [~,~,~,~,runtime] = main(2, 'lfpw', 'performTesting', 1, 'models', models.results.models, 'showOutputs', 0);
        results.lfpw.parametric(iI) = runtime;
    end
    disp(['MEAN ERROR: ' num2str(mean(results.lfpw.parametric))]);

    clear models;
    models = load('results/initsel_2d_aflw_1.mat');
    for iI = 1:NRUNS
        disp(['2D AFLW RUN ' num2str(iI) '/' num2str(NRUNS)]);
        [~,~,~,~,runtime] = main(3, 'aflw', 'performTesting', 1, 'models', models.results.models, 'showOutputs', 0);
        results.aflw.d2d(iI) = runtime;
    end
    disp(['MEAN ERROR: ' num2str(mean(results.aflw.d2d))]);

    clear models;
    models = load('results/initsel_2d_lfpw_1.mat');
    for iI = 1:NRUNS
        disp(['2D LFPW RUN ' num2str(iI) '/' num2str(NRUNS)]);
        [~,~,~,~,runtime] = main(3, 'lfpw', 'performTesting', 1, 'models', models.results.models, 'showOutputs', 0);
        results.lfpw.d2d(iI) = runtime;
    end
    disp(['MEAN ERROR: ' num2str(mean(results.lfpw.d2d))]);

    clear models;
    models = load('results/initsel_3d_aflw_1.mat');
    for iI = 1:NRUNS
        disp(['3D AFLW RUN ' num2str(iI) '/' num2str(NRUNS)]);
        [~,~,~,~,runtime] = main(4, 'aflw', 'performTesting', 1, 'models', models.results.models, 'showOutputs', 0);
        results.aflw.d3d(iI) = runtime;
    end
    disp(['MEAN ERROR: ' num2str(mean(results.aflw.d3d))]);

    clear models;
    models = load('results/initsel_3d_lfpw_1.mat');
    for iI = 1:NRUNS
        disp(['3D LFPW RUN ' num2str(iI) '/' num2str(NRUNS)]);
        [~,~,~,~,runtime] = main(4, 'lfpw', 'performTesting', 1, 'models', models.results.models, 'showOutputs', 0);
        results.lfpw.d3d(iI) = runtime;
    end
    disp(['MEAN ERROR: ' num2str(mean(results.lfpw.d3d))]);
end
