function plotParameterSelectionInitsResults(rdata)
    V_INITS = [1 3 5 7 9 11];
    
    %% --------------------------------------------------------------------
    %  -- PREPROCESS RESULTS
    %  --------------------------------------------------------------------
    
    if nargin < 1
        % List result files
        lsfiles = dir('../results/initsel_*.mat');
        
        tstruct = struct( ...
            'numInits', 0, ...
            'runtime',  0, ...
            'shapes',   [], ...
            'poses',    [], ...
            'errors',   [] ...
        );
        
        % Prepare merging structure
        rdata = struct( ...
            'aflw', struct( ...
                'sdm',          repmat(tstruct, [length(V_INITS) 1]), ...
                'parametric',   repmat(tstruct, [length(V_INITS) 1]), ...
                'd2d',          repmat(tstruct, [length(V_INITS) 1]), ...
                'd3d',          repmat(tstruct, [length(V_INITS) 1]) ...
            ), ...
            'lfpw', struct( ...
                'sdm',          repmat(tstruct, [length(V_INITS) 1]), ...
                'parametric',   repmat(tstruct, [length(V_INITS) 1]), ...
                'd2d',          repmat(tstruct, [length(V_INITS) 1]), ...
                'd3d',          repmat(tstruct, [length(V_INITS) 1]) ...
            ) ...
        );
    
        for i = 1:length(lsfiles)
            clear tdata;
            tdata = load(['../results/' lsfiles(i).name]);
            tdata = tdata.results;
            
            tstruc = struct( ...
                'numInits', tdata.numInits, ...
                'runtime',  tdata.runtime, ...
                'shapes',   tdata.shapes, ...
                'poses',    tdata.poses, ...
                'errors',   tdata.errors ...
            );
            
            iI = find(V_INITS == tdata.numInits);
            if strcmp('aflw', tdata.dataset)
                if strcmp('sdm', tdata.algorithm)
                    rdata.aflw.sdm(iI) = tstruc;
                elseif strcmp('parametric', tdata.algorithm)
                    rdata.aflw.parametric(iI) = tstruc;
                elseif strcmp('2d', tdata.algorithm)
                    rdata.aflw.d2d(iI) = tstruc;
                elseif strcmp('3d', tdata.algorithm)
                    rdata.aflw.d3d(iI) = tstruc;
                end
            elseif strcmp('lfpw', tdata.dataset)
                if strcmp('sdm', tdata.algorithm)
                    rdata.lfpw.sdm(iI) = tstruc;
                elseif strcmp('parametric', tdata.algorithm)
                    rdata.lfpw.parametric(iI) = tstruc;
                elseif strcmp('2d', tdata.algorithm)
                    rdata.lfpw.d2d(iI) = tstruc;
                elseif strcmp('3d', tdata.algorithm)
                    rdata.lfpw.d3d(iI) = tstruc;
                end
            end
        end
        
        % Save processed initializations selection results
        save('../results/processed_initsel.mat', 'rdata');
    end
    
    %% --------------------------------------------------------------------
    %  -- PLOT NUBER OF INITIALIZATIONS AS A LINE PER ALGORITHM, ONE PLOT
    %  -- PER DATASET.
    %  --------------------------------------------------------------------
    
    function [hfig1, hfig2] = plotInitializations(data, caption)
        x_vals = zeros(length(V_INITS), 4);
        y_vals = zeros(length(V_INITS), 4);
        b_vals = zeros(length(V_INITS), 2, 4);
        
        % Prepare error vectors
        for i = 1:length(V_INITS)
            x_vals(i,[1 2 3 4]) = V_INITS(i);
            y_vals(i,:) = [mean(data.sdm(i).errors) mean(data.parametric(i).errors) mean(data.d2d(i).errors) mean(data.d3d(i).errors)];
            b_vals(i,[1 2],:) = repmat([ ...
                1.96 * std(data.sdm(i).errors) / sqrt(length(data.sdm(i).errors)) ...
                1.96 * std(data.parametric(i).errors) / sqrt(length(data.sdm(i).errors)) ...
                1.96 * std(data.d2d(i).errors) / sqrt(length(data.sdm(i).errors)) ...
                1.96 * std(data.d3d(i).errors) / sqrt(length(data.sdm(i).errors)) ...
            ], [2 1]);
        end
        
        % Plot curves (2d)
        hfig1 = figure('name', [caption ' (2D)']);
        hlns  = boundedline(x_vals(:,[1 2]), y_vals(:,[1 2]), b_vals(:,:,[1 2]), 'transparency', 0.2);
        
        hleg = legend(hlns, {'SDM', 'Parametric'});
        set(hleg, 'Location', 'northeast');
        set(gca, 'XGrid', 'on', 'YGrid', 'on');
        ylabel('Normalized mean euclidean error');
        xlabel('Num. validation initializations');
        
        % Plot curves (3d)
        hfig2 = figure('name', [caption ' (3D)']);
        hlns  = boundedline(x_vals(:,[3 4]), y_vals(:,[3 4]), b_vals(:,:,[3 4]), 'transparency', 0.2);
        
        hleg = legend(hlns, {'3D alignment', '3D regression'});
        set(hleg, 'Location', 'northeast');
        set(gca, 'XGrid', 'on', 'YGrid', 'on');
        ylabel('Normalized mean euclidean error');
        xlabel('Num. validation initializations');
    end

    plotInitializations(rdata.aflw, 'AFLW initializations selection');
    plotInitializations(rdata.lfpw, 'LFPW initializations selection');
    
    %% --------------------------------------------------------------------
    %  -- PLOT LATEX TABLES FOR THE INITIALIZATIONS SELECTION
    %  --------------------------------------------------------------------
    
    function printInitializationsLine(data, rowname)
        tstr = ['\lcell ' rowname ' '];
        for iI = 1:4
            merr = mean(data(iI).errors);
            cerr = 1.96 * std(data(iI).errors) / sqrt(length(data(iI).errors));
            tstr = [tstr '& $' num2str(merr,'%.4f') ' \pm ' num2str(cerr,'%.4f') '$ '];
        end
        disp([tstr '\\ \hhline{{1}{|~}*{' num2str(length(data)+1) '}{|-}}']);
    end

    disp(':::: AFLW DATASET ::::');
    disp('\begin{tabular}{|r|c|c|c|c|}');
    disp('\hhline{~----}');
    disp('\multicolumn{1}{c|}{} & SDM & Parametric & 3D alignment & 3D regression \\ \hhline{{1}{|~}*{5}{|-}}');
    printInitializationsLine([rdata.aflw.sdm(1) rdata.aflw.parametric(1) rdata.aflw.d2d(1) rdata.aflw.d3d(1)], '1');
    printInitializationsLine([rdata.aflw.sdm(2) rdata.aflw.parametric(2) rdata.aflw.d2d(2) rdata.aflw.d3d(2)], '3');
    printInitializationsLine([rdata.aflw.sdm(3) rdata.aflw.parametric(3) rdata.aflw.d2d(3) rdata.aflw.d3d(3)], '5');
    printInitializationsLine([rdata.aflw.sdm(4) rdata.aflw.parametric(4) rdata.aflw.d2d(4) rdata.aflw.d3d(4)], '7');
    printInitializationsLine([rdata.aflw.sdm(5) rdata.aflw.parametric(5) rdata.aflw.d2d(5) rdata.aflw.d3d(5)], '8');
    printInitializationsLine([rdata.aflw.sdm(6) rdata.aflw.parametric(6) rdata.aflw.d2d(6) rdata.aflw.d3d(6)], '11');
    disp('\end{tabular}')
    disp(' ');
    

    disp(':::: LFPW DATASET ::::');
    disp('\begin{tabular}{|r|c|c|c|c|}');
    disp('\hhline{~----}');
    disp('\multicolumn{1}{c|}{} & SDM & Parametric & 3D alignment & 3D regression \\ \hhline{{1}{|~}*{5}{|-}}');
    printInitializationsLine([rdata.lfpw.sdm(1) rdata.lfpw.parametric(1) rdata.lfpw.d2d(1) rdata.lfpw.d3d(1)], '1');
    printInitializationsLine([rdata.lfpw.sdm(2) rdata.lfpw.parametric(2) rdata.lfpw.d2d(2) rdata.lfpw.d3d(2)], '3');
    printInitializationsLine([rdata.lfpw.sdm(3) rdata.lfpw.parametric(3) rdata.lfpw.d2d(3) rdata.lfpw.d3d(3)], '5');
    printInitializationsLine([rdata.lfpw.sdm(4) rdata.lfpw.parametric(4) rdata.lfpw.d2d(4) rdata.lfpw.d3d(4)], '7');
    printInitializationsLine([rdata.lfpw.sdm(5) rdata.lfpw.parametric(5) rdata.lfpw.d2d(5) rdata.lfpw.d3d(5)], '8');
    printInitializationsLine([rdata.lfpw.sdm(6) rdata.lfpw.parametric(6) rdata.lfpw.d2d(6) rdata.lfpw.d3d(6)], '11');
    disp('\end{tabular}')
    disp(' ');
end