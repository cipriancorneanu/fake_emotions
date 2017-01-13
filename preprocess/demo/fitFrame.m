function [shape, pose] = fitFrame(img, rect, faceModel2d)
    addpath(genpath('../sdm'));
    
    timg = zeros(rect(3:4), 'uint8');
    sca = rect(3) / 200;
    res = size(img);
    
    trect = [rect(1:2) rect(1:2)+rect(3:4)-1];
    iBox = [max(2-trect(1:2), [1 1]) min(trect(3:4), res(1:2))-trect(1:2)+1];
    trect = [max(trect(1:2), 1) min(trect(3:4), res(1:2))];

    timg(iBox(2):iBox(4), iBox(1):iBox(3)) = rgb2gray(img(trect(2):trect(4), trect(1):trect(3), :));

    [shape, pose] = algorithmSDMTest_pose_simple(faceModel2d,...
                        {im2double(imresize(timg, [200 200]))}, ...
                        'normIndexs', [17 18], ...
                        'showOutputs', 0, ...
                        'initType', 'random' ...
                    );
                
    lsShapes = zeros(length(shape),2,2);
    lsPoses  = repmat(struct('roll',0,'pitch',0,'yaw',0), [1 4]);
    
    shape = bsxfun(@plus, shape(:,1:2)*sca, rect(1:2));
    lsShapes(:,:,1:end-1) = lsShapes(:,:,2:end);
    lsShapes(:,:,end) = shape;
    shape = mean(lsShapes(:, :, squeeze(sum(sum(lsShapes,1),2) ~= 0)), 3);
    lsPoses(1:end-1) = lsPoses(2:end);
end

