function [dataset] = relabeler(dataset, fsave)
    function evtclick(src, evt)
        if strcmp(evt.Key, 'f') == 1, finish = true; end
        if strcmp(evt.Key, 'c') == 1
            iL = size(landmarks,1)+1;
            captured = true;
        end
    end

    iL = 0;
    function evtdown(src, evt)
        wP = get(src, 'Position');
        aP = get(get(src, 'CurrentAxes'), 'Position') .* wP([3 4 3 4]);

        loc = (get(hdlr,'CurrentPoint') - aP(1:2)) * 200 ./ aP(3:4);
        loc(2) = 200 - loc(2);
        
        if captured == false
            [~,iL] = min(sum(bsxfun(@minus, landmarks, loc).^2, 2));
            landmarks(iL,:) = [0 0];
            captured = true;
        else
            edit = true;
            landmarks(iL,:) = loc;
            captured = false;
        end
        
        refreshLandmarks(landmarks);
        drawnow;
    end

    function refreshLandmarks(landmarks)
        set(hLmkG,  'XData', landmarks(1:8,1), 'YData', landmarks(1:8,2));
        set(hLmkC,  'XData', landmarks([9:16 22],1), 'YData', landmarks([9:16 22],2));
        set(hLmkR,  'XData', landmarks([17:18 21],1), 'YData', landmarks([17:18 21],2));
        set(hLmkP,  'XData', landmarks(19:20,1), 'YData', landmarks(19:20,2));
        set(hLmkUM, 'XData', landmarks(25:26,1), 'YData', landmarks(25:26,2));
        set(hLmkBM, 'XData', landmarks(27:28,1), 'YData', landmarks(27:28,2));
        set(hLmkB,  'XData', landmarks([23 24 29:end],1), 'YData', landmarks([23 24 29:end],2));
    end

    % Prepare interface
    hdlr = figure('WindowButtonDownFcn', @evtdown, 'KeyPressFcn', @evtclick);
    set(gca,'YDir','reverse', 'xtick', [], 'ytick', [], 'XLim', [0 200], 'YLim', [0 200]);
    hold on;
    hImg = image(uint8(zeros(200, 200, 3)));
    hLmkG  = scatter([5 12], [6 9], 'green');
    hLmkC  = scatter([5 12], [6 9], 'cyan');
    hLmkR  = scatter([5 12], [6 9], 'red');
    hLmkP  = scatter([5 12], [6 9], 'yellow');
    hLmkUM = scatter([5 12], [6 9], 'magenta');
    hLmkBM = scatter([5 12], [6 9], 'green');
    hLmkB  = scatter([6 10], [6 7], 'blue');
    hold off;

    edit = false;
    for i = 1:length(dataset)
        landmarks = dataset(i).landmarks2d;
        
        set(hdlr, 'Name', [num2str(i) '/' num2str(length(dataset))]);
        set(hImg, 'cdata', uint8(repmat(dataset(i).face*255, [1 1 3])));
        refreshLandmarks(landmarks);
        drawnow;
        
        captured = false;
        finish = false;
        while ~finish
            pause(0.01);
        end
        
        dataset(i).landmarks2d = landmarks;
        if (mod(i, 25) == 0) && (edit == true)
            save(fsave, 'dataset');
            edit = false;
        end
    end
    
    save(fsave, 'dataset');
end