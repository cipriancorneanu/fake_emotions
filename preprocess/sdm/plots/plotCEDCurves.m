function [hfig1,hfig2] = plotCEDCurves(errsdm, errparam, err2d, err3d, caption)
    function [hline] = plotCEDCurve(errors)
        xvals = [0 ; sort(errors, 'ascend')];
        yvals = [(0:length(xvals)-1) / length(xvals)];
        hline = plot(xvals, yvals);
    end
    
    hfig1 = figure('name', [caption ' (2D)']);
    hold on;
    
    set(plotCEDCurve(errsdm), 'color', [0 0 1]);
    set(plotCEDCurve(errparam), 'color', [1 0 0]);
    
    hleg = legend('SDM', 'parametric');
    set(hleg, 'Location', 'southeast');
    set(gca, 'XGrid', 'on', 'YGrid', 'on');
    xlabel('Normalized mean euclidean error');
    ylabel('Cumulative rate of samples');
    xlim([0 0.4]);
    
    hold off;
    
    
    hfig2 = figure('name', [caption ' (3D)']);
    hold on;
    
    set(plotCEDCurve(err2d), 'color', [0 0 1]);
    set(plotCEDCurve(err3d), 'color', [1 0 0]);
    
    hleg = legend('3D alignment', '3D regression');
    set(hleg, 'Location', 'southeast');
    set(gca, 'XGrid', 'on', 'YGrid', 'on');
    xlabel('Normalized mean euclidean error');
    ylabel('Cumulative rate of samples');
    xlim([0 0.4]);
    
    hold off;
end