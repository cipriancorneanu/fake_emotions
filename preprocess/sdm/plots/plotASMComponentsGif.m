function plotASMComponentsGif(asm)
    imsz = [420 560];
    
    mean = reshape(asm.mean, 2, [])' * [1 0 ; 0 -1];
    pcs  = {reshape(asm.transform(2,:), 2, [])' * [1 0 ; 0 -1] reshape(asm.transform(3,:), 2, [])' * [1 0 ; 0 -1]};
    evs  = sqrt(asm.eigenvalues(1:2));
    
    function [frame] = drawFrame(fr)
        hfig = figure('Color', [1 1 1]);
        hold on;

        % Left eye
        fill([fr(09,01) fr(13,01) fr(17,01)], [fr(09,02) fr(13,02) fr(17,02)], [1.0 1.0 1.0], 'EdgeColor', 'None');
        fill([fr(11,01) fr(13,01) fr(17,01)], [fr(11,02) fr(13,02) fr(17,02)], [0.95 0.95 0.95], 'EdgeColor', 'None');
        fill([fr(09,01) fr(14,01) fr(17,01)], [fr(09,02) fr(14,02) fr(17,02)], [0.95 0.95 0.95], 'EdgeColor', 'None');
        fill([fr(11,01) fr(14,01) fr(17,01)], [fr(11,02) fr(14,02) fr(17,02)], [0.9 0.9 0.9], 'EdgeColor', 'None');
        plot([fr(09,01) fr(13,01) fr(11,01), fr(14,01) fr(09,01)], [fr(09,02) fr(13,02) fr(11,02), fr(14,02) fr(09,02)], 'Color', [.5 .5 .5]);

        % Right eye
        fill([fr(12,01) fr(15,01) fr(18,01)], [fr(12,02) fr(15,02) fr(18,02)], [1.0 1.0 1.0], 'EdgeColor', 'None');
        fill([fr(10,01) fr(15,01) fr(18,01)], [fr(10,02) fr(15,02) fr(18,02)], [0.95 0.95 0.95], 'EdgeColor', 'None');
        fill([fr(12,01) fr(16,01) fr(18,01)], [fr(12,02) fr(16,02) fr(18,02)], [0.95 0.95 0.95], 'EdgeColor', 'None');
        fill([fr(10,01) fr(16,01) fr(18,01)], [fr(10,02) fr(16,02) fr(18,02)], [0.9 0.9 0.9], 'EdgeColor', 'None');
        plot([fr(12,01) fr(15,01) fr(10,01), fr(16,01) fr(12,01)], [fr(12,02) fr(15,02) fr(10,02), fr(16,02) fr(12,02)], 'Color', [.5 .5 .5]);

        % Eye centres
        scatter(fr([17 18],1), fr([17 18],2), 50, [0 0 0]);

        % Left eyebrow
        fill( ...
            [fr(01,01) fr(05,01) fr(03,01) fr(06,01)], ...
            [fr(01,02) fr(05,02) fr(03,02) fr(06,02)], ...
            [0.2 0.2 0.2] ...
         );

        % Right eyebrow
        fill( ...
            [fr(04,01) fr(07,01) fr(02,01) fr(08,01)], ...
            [fr(04,02) fr(07,02) fr(02,02) fr(08,02)], ...
            [0.2 0.2 0.2] ...
         );
        
        % Nose
        fill([fr(19,01) fr(21,01) fr(22,01)], [fr(19,02) fr(21,02) fr(22,02)], [0.6 0.5 0.5]);
        fill([fr(20,01) fr(21,01) fr(22,01)], [fr(20,02) fr(21,02) fr(22,02)], [0.5 0.4 0.4]);

        % Mouth
        fill([fr(23,01) fr(25,01) fr(26,01)], [fr(23,02) fr(25,02) fr(26,02)], [0.9 0.7 0.7]);
        fill([fr(24,01) fr(25,01) fr(26,01)], [fr(24,02) fr(25,02) fr(26,02)], [0.85 0.65 0.65]);
        fill([fr(23,01) fr(27,01) fr(28,01)], [fr(23,02) fr(27,02) fr(28,02)], [0.9 0.7 0.7]);
        fill([fr(24,01) fr(27,01) fr(28,01)], [fr(24,02) fr(27,02) fr(28,02)], [0.85 0.65 0.65]);

        % Chin
        scatter(fr(29,1), fr(29,2), 50, [0 0 0], 'fill');

        hold off;
        xlim([-70 70]);
        ylim([-110 60]);
        axis off;
        
        frame = frame2im(getframe(gca));
        frame = imresize(frame, imsz);
        close(hfig);
    end

    function saveGIF(fname, frames)
        for iF = 1:size(frames, 4)
            frame = frames(:,:,:,iF);
            [imind,cm] = rgb2ind(frame,256);
            if iF == 1
                imwrite(imind, cm, fname, 'gif', 'DelayTime', 0, 'loopcount', inf);
            else
                imwrite(imind, cm, fname, 'gif', 'DelayTime', 0, 'writemode', 'append');
            end
        end
    end

    function savePNGs(fprefix, frames)
        for iF = 1:size(frames, 4)
            frame = squeeze(frames(:,:,:,iF));
            imwrite(frame, [fprefix num2str(iF-1) '.png'], 'png');
        end
    end

    frames = zeros(imsz(1), imsz(2), 3, 1, 'uint8');
    
    find = 1;
    for pc = 1:2
        for w = 0:(5/20):2.5
            fr = mean + pcs{pc}*evs(pc)*w;
            frames(:,:,:,find) = drawFrame(fr);
            find = find+1;
        end

        for w = 2.5:(-5/20):-2.5
            fr = mean + pcs{pc}*evs(pc)*w;
            frames(:,:,:,find) = drawFrame(fr);
            find = find+1;
        end
        
        for w = -2.5:(5/20):0
            fr = mean + pcs{pc}*evs(pc)*w;
            frames(:,:,:,find) = drawFrame(fr);
            find = find+1;
        end
    end
    
    % Save GIF image
    %saveGIF('pcabases.gif', frames);
    
    % Save PNG images
    savePNGs('pcabases_', frames);
    
    %implay(frames);
end