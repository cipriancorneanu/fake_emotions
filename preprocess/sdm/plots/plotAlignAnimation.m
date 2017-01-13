function plotAlignAnimation(ASM3D, shape, img, caption)
    addpath(genpath('../libs'))
    
    ctr = 0;
    function captureShape(Data3D)
        set(hAli, 'XData', Data3D(:,1), 'YData', Data3D(:,2));
        drawnow;
        
        imwrite(frame2im(getframe(gca)), ['align3d_' num2str(ctr) '.png']);
        ctr = ctr + 1;
        pause(1);
    end
    
    function [fModel, fWeights, fValues, shpTrack] = AnimateAlign(ASM3D, Data2D)
        nBases = length(ASM3D.eigenvalues);
        w = zeros(nBases, 1);
        Data2D(:,end+1) = 1;

        % Set algorithm parameters
        iNor = [17 18];

        % Prepare transformed PCA mean and bases
        meanPCA = reshape(ASM3D.mean, 3, []);
        basesPCA = zeros([size(meanPCA) nBases]);
        for i = 1:nBases
            basesPCA(:,:,i) = reshape(ASM3D.transform(i,:), 3, []);
        end

        % Prepare error tracking variables
        mae = [];
        shpTrack = [];
        normFac = sqrt(sum((Data2D(iNor(1),:) - Data2D(iNor(2),:)) .^ 2));

        Data3D = reshape(ASM3D.mean(:) + ASM3D.transform' * w, 3, [])';
        Data3D = bsxfun(@plus, Data3D(:,1:2), [99 76]);
        captureShape(Data3D);
        
        for it = 1:5
            % Generate current 3D shape
            Data3D = reshape(ASM3D.mean(:) + ASM3D.transform' * w, 3, [])';
            [amodel, newV] = align3D(Data2D(:,1:2)', Data3D');

            % Get alignment/rotation+scaling projection transforms
            tfm = amodel.projection(1:2,1:3);
            off = amodel.projection(1:2,4);
            rsc = tfm(1:2,1:3);
            
            Data3D = bsxfun(@plus, tfm * Data3D', off)';
            captureShape(Data3D);

            % Calculate current error and stop if converged
            mae(it) = mean(sqrt(sum((newV(1:2,:)' - Data2D(:,1:2)) .^ 2, 2))) / normFac;
            if it>1 && mae(it) >= mae(it-1)
                break;
            end

            % Save current model
            fModel   = amodel;
            fWeights = w;
            fValues  = newV;

            % Project pca mean and bases
            prjMean = bsxfun(@plus, tfm * meanPCA, off);
            prjMean = reshape(prjMean(1:2,:), [], 1);
            prjBases = zeros(length(prjMean), nBases);
            for i = 1:nBases
                %tmp = bsxfun(@plus, tfm * basesPCA(:,:,i), off);
                tmp = rsc * basesPCA(:,:,i);
                prjBases(:,i) = reshape(tmp, [], 1);
            end

            % Optimize shape coefficients
            targ = reshape(Data2D(:,1:2)', [], 1) - prjMean;
            w = rescalePCAOutliers((prjBases \ targ)', ASM3D.eigenvalues)';
            
            Data3D = reshape(ASM3D.mean(:) + ASM3D.transform' * w, 3, [])';
            Data3D = bsxfun(@plus, tfm * Data3D', off)';
            captureShape(Data3D);
        end
    end

    % Prepare interface
    hFig = figure('name', caption);
    axis([0 200 0 200]);
    set(gca,'YDir','reverse', 'xtick', [], 'ytick', []);
    hold on;
    hImg = image(repmat(uint8(img*255), [1 1 3]));
    hShp = scatter(shape(:,1), shape(:,2), 'red');
    hAli = scatter(shape(:,1), shape(:,2), 'cyan');
    hold off;
    
    AnimateAlign(ASM3D, shape);
    close(hFig);
end