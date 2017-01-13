function [] = plotPoseFittingResults(results)
    V_PITCH  = [-90 -75 -60 -45 -30 -15 0 +15 +30 +45 +60 +75 +90] * 0.75;
    V_YAW    = [-90 -75 -60 -45 -30 -15 0 +15 +30 +45 +60 +75 +90];
    
    % Load data
    if nargin == 0
        results = load('../results/processed_posefit.mat');
        results = results.results;
    end
    
    %% --------------------------------------------------------------------
    %  -- PLOT PITCH AND YAW ERROR SURFACES AND LATEX TABLES
    %  --------------------------------------------------------------------
    
    function [hpitch, hyaw] = plotPoseErrorSurfacesAndTables(sdata, caption)
        sdata_pitch = zeros(length(V_PITCH), length(V_YAW));
        sdata_yaw   = zeros(length(V_PITCH), length(V_YAW));
        for iP = 1:length(V_PITCH)
            for iY = 1:length(V_YAW)
                sdata_pitch(iP,iY) = mean(sdata(iP,iY).errors_pitch);
                sdata_yaw(iP,iY)   = mean(sdata(iP,iY).errors_yaw);
            end
        end
        
        function [hdlr] = plotPoseErrorSurface(sdata, caption)
            % Plot error surface plot (pitch)
            sdata(isnan(sdata)) = 0;
            fdata = zeros(size(sdata,1)+1, size(sdata,2)+1);
            fdata(1:end-1,1:end-1) = sdata;
            
            figure('name', caption);
            hdlr = pcolor([V_YAW-7.5 97.5], [V_PITCH-5.625 73.125], fdata);
            caxis([0 70]);
            set(gca, 'YTick', V_PITCH, 'XTick', V_YAW);
            ylabel('Pitch');
            xlabel('Yaw');
            colorbar;
        end
        
        function printPoseErrorTable(sdata, caption)
            % Print table title
            disp(['Table: ' caption]);

            % Prepare align characters
            aligns = 'r';
            hline  = '~';
            tstr = '\multicolumn{1}{c|}{} ';
            for i = 1:length(V_YAW)
                aligns = [aligns 'c'];
                hline = [hline '-'];
                tstr = [tstr '& \lcell $' num2str(V_YAW(i)) '$ '];
            end

            disp(['\begin{tabular}{' aligns '}']);
            disp(['\hhline{' hline '}']);
            disp([tstr '\\ \hhline{{1}{|~}*{' num2str(length(V_YAW)+1) '}{|-}}']);
            for iP = 1:length(V_PITCH)
                tstr = ['\lcell $' num2str(V_PITCH(iP)) '$ '];
                for iY = 1:length(V_YAW)
                    if isnan(sdata(iP,iY))
                        tstr = [tstr '& - '];
                    else
                        tstr = [tstr '& $' num2str(sdata(iP,iY), '%.2f') '$ '];
                    end
                end
                disp([tstr '\\ \hhline{{1}{|~}*{' num2str(length(V_YAW)+1) '}{|-}}']);
            end
            disp('\end{tabular}');
            disp(' ');
        end
        
        % Plot error surface plots
        hpitch = plotPoseErrorSurface(sdata_pitch, [caption ' (pitch)']);
        hyaw   = plotPoseErrorSurface(sdata_yaw, [caption ' (yaw)']);
        
        % Print error tables
        printPoseErrorTable(sdata_pitch, [caption ' (pitch)']);
        printPoseErrorTable(sdata_yaw, [caption ' (yaw)']);
    end 
    
    plotPoseErrorSurfacesAndTables(results.d2d, '2D fitting pose error');
    plotPoseErrorSurfacesAndTables(results.d3d, '3D fitting pose error');
end