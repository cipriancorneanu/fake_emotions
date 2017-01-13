    function [] = plotResults(images, shapes)
        [nsL,nsD,nI] = size(shapes);
        f = figure;
        
        nI = min(nI, 1);
        for k = 1:nI
            %subplot(1,1,k);
            %hold on;
            imshow(images{k}); hold on;
            plot(shapes(:,1,k), shapes(:,2,k), 'g.',...
                'MarkerSize',10);
            hold off;
        end
        
        uiwait(f); 
    end