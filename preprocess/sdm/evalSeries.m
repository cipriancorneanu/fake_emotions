function [results] = evalSeries(results)
    errors_pitch = zeros(1,30); 
    errors_yaw = zeros(1,30); 
    counts = zeros(1,30);
    labels = zeros(1,30);
    lidx = 0;
    
    d2dtotal_counts = 0;
    d2dtotal_pitch_errors = 0;
    d2dtotal_yaw_errors = 0;
    d3dtotal_counts = 0;
    d3dtotal_pitch_errors = 0;
    d3dtotal_yaw_errors = 0;
    
    for p=1:13
        for y=1:13
            series = results.d3d(p,y).series;
            for i=1:length(series)
                idx = find(labels == series(i));
                if isempty(idx)
                    lidx = lidx+1;
                    idx = lidx;
                    labels(idx) = series(i);
                end
                
                errors_pitch(idx) = errors_pitch(idx) + results.d3d(p,y).errors_pitch(i);
                errors_yaw(idx)   = errors_yaw(idx) + results.d3d(p,y).errors_yaw(i);
                counts(idx)       = counts(idx) + 1;
                
                d2dtotal_counts = d2dtotal_counts + 1;
                d2dtotal_pitch_errors = [d2dtotal_pitch_errors results.d2d(p,y).errors_pitch(i)];
                d2dtotal_yaw_errors = [d2dtotal_yaw_errors results.d2d(p,y).errors_yaw(i)];
                d3dtotal_counts = d3dtotal_counts + 1;
                d3dtotal_pitch_errors = [d3dtotal_pitch_errors results.d3d(p,y).errors_pitch(i)];
                d3dtotal_yaw_errors = [d3dtotal_yaw_errors results.d3d(p,y).errors_yaw(i)];
            end
        end
    end
    
    disp(['d2d pitch error: ' num2str(mean(d2dtotal_pitch_errors)) ' \pm ' num2str(1.96 * std(d2dtotal_pitch_errors) / sqrt(length(d2dtotal_pitch_errors)))]);
    disp(['d2d yaw error: ' num2str(mean(d2dtotal_yaw_errors)) ' \pm ' num2str(1.96 * std(d2dtotal_yaw_errors) / sqrt(length(d2dtotal_yaw_errors)))]);
    disp(['d3d pitch error: ' num2str(mean(d3dtotal_pitch_errors)) ' \pm ' num2str(1.96 * std(d3dtotal_pitch_errors) / sqrt(length(d3dtotal_pitch_errors)))]);
    disp(['d3d yaw error: ' num2str(mean(d3dtotal_yaw_errors)) ' \pm ' num2str(1.96 * std(d3dtotal_yaw_errors) / sqrt(length(d3dtotal_yaw_errors)))]);
    
    [errs, inds] = sort((errors_pitch + errors_yaw) ./ counts, 'descend');
    rlabels = labels(inds(1:4));
    
    for p=1:13
        for y=1:13
            indexs = sort(find(ismember(results.d3d(p,y).series(2:end), rlabels)), 'descend');
            for i = indexs
                results.d2d(p,y).series(i) = [];
                results.d2d(p,y).shapes(:,:,i) = [];
                results.d2d(p,y).poses(i) = [];
                results.d2d(p,y).errors_pitch(i) = [];
                results.d2d(p,y).errors_yaw(i) = [];
                
                results.d3d(p,y).series(i) = [];
                results.d3d(p,y).shapes(:,:,i) = [];
                results.d3d(p,y).poses(i) = [];
                results.d3d(p,y).errors_pitch(i) = [];
                results.d3d(p,y).errors_yaw(i) = [];
            end
        end
    end
end