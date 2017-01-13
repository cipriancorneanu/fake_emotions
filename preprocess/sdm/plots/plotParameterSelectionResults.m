function [] = plotParameterSelectionResults(rdata)
    V_BOOSTS = [1 5 10 15 20 25];
    V_STEPS  = [1 2 3 4 5 6 7];

    addpath(genpath('../libs'));
    
    %% --------------------------------------------------------------------
    %  -- PREPROCESS RESULTS
    %  --------------------------------------------------------------------
    
    if nargin < 1
        % List result files
        lsfiles = dir('../results/parsel_*.mat');

        % Prepare stub structure
        trstruc = struct( ...
            'numCascadeSteps',  0, ...
            'numBoosts',        0, ...
            'runtime',          0, ...
            'shapes',           [], ...
            'poses',            [], ...
            'errors',           [] ...
        );
        
        % Prepare merging structure
        rdata = struct( ...
            'aflw', struct( ...
                'sdm',          repmat(trstruc, length(V_BOOSTS), length(V_STEPS)), ...
                'parametric',   repmat(trstruc, length(V_BOOSTS), length(V_STEPS)), ...
                'd2d',          repmat(trstruc, length(V_BOOSTS), length(V_STEPS)), ...
                'd3d',          repmat(trstruc, length(V_BOOSTS), length(V_STEPS))  ...
            ), ...
            'lfpw', struct( ...
                'sdm',          repmat(trstruc, length(V_BOOSTS), length(V_STEPS)), ...
                'parametric',   repmat(trstruc, length(V_BOOSTS), length(V_STEPS)), ...
                'd2d',          repmat(trstruc, length(V_BOOSTS), length(V_STEPS)), ...
                'd3d',          repmat(trstruc, length(V_BOOSTS), length(V_STEPS))  ...
            ) ...
        );

        % Merge results into a single structure
        for i = 1:length(lsfiles)
            tdata = load(['../results/' lsfiles(i).name]);
            tdata = tdata.results;
            
            trstruc = struct( ...
                'numCascadeSteps',  tdata.numCascadeSteps, ...
                'numBoosts',        tdata.numBoosts, ...
                'runtime',          tdata.runtime, ...
                'shapes',           tdata.shapes, ...
                'poses',            tdata.poses, ...
                'errors',           tdata.errors ...
            );

            iB = find(V_BOOSTS == trstruc.numBoosts);
            iS = find(V_STEPS == trstruc.numCascadeSteps);
                
            if strcmp('aflw', tdata.dataset)
                if tdata.algorithm == 1
                    rdata.aflw.sdm(iB,iS) = trstruc;
                elseif tdata.algorithm == 2
                    rdata.aflw.parametric(iB,iS) = trstruc;
                elseif tdata.algorithm == 3
                    rdata.aflw.d2d(iB,iS) = trstruc;
                elseif tdata.algorithm == 4
                    rdata.aflw.d3d(iB,iS) = trstruc;
                end
            elseif strcmp('lfpw', tdata.dataset)
                if tdata.algorithm == 1
                    rdata.lfpw.sdm(iB,iS) = trstruc;
                elseif tdata.algorithm == 2
                    rdata.lfpw.parametric(iB,iS) = trstruc;
                elseif tdata.algorithm == 3
                    rdata.lfpw.d2d(iB,iS) = trstruc;
                elseif tdata.algorithm == 4
                    rdata.lfpw.d3d(iB,iS) = trstruc;
                end
            end
        end

        % Save processed parameter selection results
        save('../results/processed_parsel.mat', 'rdata');
    end
    
    %% --------------------------------------------------------------------
    %  -- GENERATE PLOTS FOR THE BOOSTS/STEPS PARAMETERS OVER BOTH 
    %  -- ALIGNMENT DATASETS AND ALL ALGORITHMS.
    %  --------------------------------------------------------------------
    
    function [hfig] = plotErrorGridSearch(data, caption)
        x_vals = zeros(length(V_STEPS), length(V_BOOSTS));
        y_vals = zeros(length(V_STEPS), length(V_BOOSTS));
        u_vals = zeros(length(V_STEPS), length(V_BOOSTS));
        l_vals = zeros(length(V_STEPS), length(V_BOOSTS));
        
        labels = cell(1, length(V_BOOSTS));
        for iB = 1:length(V_BOOSTS)
            for iS = 1:length(V_STEPS)
                x_vals(iS,iB)   = V_STEPS(iS);
                y_vals(iS,iB)   = mean(data(iB,iS).errors);
                u_vals(iS,iB)   = 1.96 * std(data(iB,iS).errors) / sqrt(length(data(iB,iS).errors));
                l_vals(iS,iB)   = 1.96 * std(data(iB,iS).errors) / sqrt(length(data(iB,iS).errors));
            end
            
            labels{iB} = [num2str(V_BOOSTS(iB)) ' data augm.'];
        end
        
        %boundedline(x_vals, y_vals, b_vals);
        hfig = figure('name', caption);
        hold on;
        errorbar(x_vals, y_vals, u_vals, l_vals);
        ymin = min(y_vals(:));
        ymax = max(y_vals(:));
        
        hleg = legend(labels{:});
        set(hleg, 'Location', 'northeast');
        set(gca, 'XGrid', 'on', 'YGrid', 'on');
        xlim([min(V_STEPS) max(V_STEPS)]);
        ylim([floor(ymin*100)/100 round(ymax*100)/100]);
        xlabel('Num. cascade steps');
        ylabel('Normalized mean euclidean error');
        
        hold off;
    end

    % Plot results for the 2D methods
%     plotErrorGridSearch(rdata.aflw.sdm, 'AFLW - Supervised Descent Method');
%     plotErrorGridSearch(rdata.aflw.parametric, 'AFLW - Parametric');
%     plotErrorGridSearch(rdata.lfpw.sdm, 'LFPW - Supervised Descent Method');
%     plotErrorGridSearch(rdata.lfpw.parametric, 'LFPW - Parametric');
%     
    % Plot results for the 3D methods
%     plotErrorGridSearch(rdata.aflw.d2d, 'AFLW - 2D');
%     plotErrorGridSearch(rdata.aflw.d3d, 'AFLW - 3D');
    plotErrorGridSearch(rdata.lfpw.d2d, 'LFPW - 2D');
    plotErrorGridSearch(rdata.lfpw.d3d, 'LFPW - 3D');
    
    %% --------------------------------------------------------------------
    %  -- GENERATE LATEX TABLES FOR THE BOOSTS/STEPS GRID SEARCH FOR BOTH
    %  -- ALIGNMENT DATASETS AND ALL ALGORITHMS.
    %  --------------------------------------------------------------------
    
    function generateLatexTable(data, caption)
        disp(['Table: ' caption]);
        
        % Prepare align characters
        aligns = 'r';
        hline  = '~';
        tstr = '\multicolumn{1}{c|}{} ';
        for i = 1:length(V_BOOSTS)
            aligns = [aligns '|c'];
            hline  = [hline '-'];
            tstr = [tstr '& \lcell ' num2str(V_BOOSTS(i)) ' '];
        end
        
        disp(['\begin{tabular}{|' aligns '|}']);
        disp(['\hhline{' hline '}']);
        disp([tstr '\\ \hhline{{1}{|~}*{' num2str(length(V_BOOSTS)+1) '}{|-}}']);
        for iS = 1:length(V_STEPS)
            tstr = ['\lcell ' num2str(V_STEPS(iS)) ' '];
            for iB = 1:length(V_BOOSTS)
                merr = mean(data(iB,iS).errors);
                cerr = 1.96 * std(data(iB,iS).errors) / sqrt(length(data(iB,iS).errors));
                tstr = [tstr '& $' num2str(merr,'%.4f') ' \pm ' num2str(cerr,'%.4f') '$ '];
            end
            disp([tstr '\\ \hhline{{1}{|~}*{' num2str(length(V_BOOSTS)+1) '}{|-}}']);
        end
        disp('\end{tabular}');
        disp(' ');
    end

    % Plot results for the 2D methods
    generateLatexTable(rdata.aflw.sdm, 'AFLW - Supervised Descent Method');
    generateLatexTable(rdata.aflw.parametric, 'AFLW - Parametric');
    generateLatexTable(rdata.lfpw.sdm, 'LFPW - Supervised Descent Method');
    generateLatexTable(rdata.lfpw.parametric, 'LFPW - Parametric');
    
    % Plot results for the 3D methods
    generateLatexTable(rdata.aflw.d2d, 'AFLW - 2D');
    generateLatexTable(rdata.aflw.d3d, 'AFLW - 3D');
    generateLatexTable(rdata.lfpw.d2d, 'LFPW - 2D');
    generateLatexTable(rdata.lfpw.d3d, 'LFPW - 3D');
end