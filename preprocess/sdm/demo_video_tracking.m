function [] = demo_video_tracking(video)

    
    %{
    while hasFrame(v)
        video = readFrame(v);
    end
    whos video
    %}
    %record = 1;
    %if ~exist('record', 'var'), record = 1; end
    %addpath(genpath('./libs'))
    
    % Configure video input
    %vid = VideoReader(video);
    %res = get(vid, 'VideoResolution');
    
    %set(vid, 'ReturnedColorSpace', 'RGB');
    %triggerconfig(vid, 'manual');
      
    % Configure record output
    aviobj = VideoWriter('demo.avi');
    aviobj.Quality = 100;
    aviobj.FrameRate = 6;
    open(aviobj);
    
    
    lsShapes = zeros(29,2,2);
    lsPoses  = repmat(struct('roll',0,'pitch',0,'yaw',0), [1 4]);
    function [shape, pose] = fitFrame(img, rect)
        timg = zeros(rect(3:4), 'uint8');
        sca = rect(3) / 200;
        
        trect = [rect(1:2) rect(1:2)+rect(3:4)-1];
        iBox = [max(2-trect(1:2), [1 1]) min(trect(3:4), res)-trect(1:2)+1];
        trect = [max(trect(1:2), 1) min(trect(3:4), res)];
        
        timg(iBox(2):iBox(4), iBox(1):iBox(3)) = rgb2gray(img(trect(2):trect(4), trect(1):trect(3), :));
        [shape,pose] = algorithmSDMTest_pose3d( ...
            faceModel2d, {im2double(imresize(timg, [200 200]))}, ...
            'showOutputs', 0, ...
            'normIndexs', [17 18] ...
        );
        
        shape = bsxfun(@plus, shape(:,1:2)*sca, rect(1:2));
        lsShapes(:,:,1:end-1) = lsShapes(:,:,2:end);
        lsShapes(:,:,end) = shape;
        shape = mean(lsShapes(:, :, squeeze(sum(sum(lsShapes,1),2) ~= 0)), 3);
        lsPoses(1:end-1) = lsPoses(2:end);
        lsPoses(end) = pose;
        
        idxs = [lsPoses.roll]+[lsPoses.pitch]+[lsPoses.yaw] ~= 0;
        pose.roll  = mean([lsPoses(idxs).roll]);
        pose.pitch = mean([lsPoses(idxs).pitch]);
        pose.yaw   = mean([lsPoses(idxs).yaw]);
    end

    function plotAxis(pose, offx, offy)
        s3 = sin(pose.yaw);   c3 = cos(pose.yaw);
        s2 = sin(pose.pitch); c2 = cos(pose.pitch);
        s1 = sin(pose.roll);  c1 = cos(pose.roll);
        rmat = [c3*c1+s3*s2*s1 c3*s1-s3*s2*c1 s3*c2 ; -c2*s1 c2*c1 s2 ; -s3*c1+c3*s2*s1 -s3*s1-c3*s2*c1 c3*c2];

        p1 = rmat * [90 0 0]';
        p2 = rmat * [0 90 0]';
        p3 = rmat * [0 0 90]';

        set(hL1, 'XData', [offx offx+p1(1)], 'YData', [offy offy+(p1(2))]);
        set(hL2, 'XData', [offx offx+p2(1)], 'YData', [offy offy+(p2(2))]);
        set(hL3, 'XData', [offx offx+p3(1)], 'YData', [offy offy+(p3(2))]);
    end

    function [rect] = calculateRect()
        shape = mean(lsShapes(:, :, squeeze(sum(sum(lsShapes,1),2) ~= 0)), 3);
        
        ul = min(shape(:,1:2), [], 1);
        bd = max(shape(:,1:2), [], 1);
        mn = (ul+bd) / 2;
        ds = max(bd-mn) * 1.5;
        
        rect = round([(mn-ds) ds*[2 2]]);
    end

    try
        % Start capturing video
        %vidRes = get(vid, 'VideoResolution');
        %start(vid);

        % Prepare detector and tracker
        faceDetector = vision.CascadeObjectDetector();
        faceModel2d    = load('model_3d.mat');
        faceModel2d    = faceModel2d.model;
        faceModel2d.initPoses = faceModel2d.initPoses(1:1,:);
        faceModel2d.initPoses(:,faceModel2d.targets.idxsRotat(1)) = [0];
        
        %{
        hdlr = figure('units','normalized','outerposition',[0 0 1 1]);
        axis([0 vidRes(1) 0 vidRes(2)]);
        set(gca,'YDir','reverse', 'xtick', [], 'ytick', []);
        hold on;
        %}
        
        v = VideoReader(video);
        res = [v.Width v.Height];
        %set(v, 'ReturnedColorSpace', 'RGB');
        
        while hasFrame(v)
            %img = rgb2gray(readFrame(v));
            img = readFrame(v);

            % Detect face
            while ~exist('rect', 'var') || isempty(rect)                    
                %img  = getsnapshot(vid);
                rect = round(step(faceDetector, img));
                rect = rect(1,:) + [0 rect(2)*0.25 0 0];        % Detect interest points
            end

            [shape, ~] = fitFrame(img, rect);
            rect = calculateRect();

            % Create figure components
            hImage = image(img);
            hRect  = rectangle('Position', rect, 'EdgeColor', 'blue', 'LineWidth', 2); %plot(rect([1 3 5 7 1]), rect([2 4 6 8 2]), 'Color', 'blue', 'LineWidth', 2);
            hShape = scatter(shape(:,1), shape(:,2), 'red');
            hL1 = line([1 2], [1 2], 'Color', 'm', 'LineWidth', 2);
            hL2 = line([1 2], [1 2], 'Color', 'c', 'LineWidth', 2);
            hL3 = line([1 2], [1 2], 'Color', 'r', 'LineWidth', 2);
            hold off;

            % Track interest points
            [shape, pose] = fitFrame(data, rect);
            rect = calculateRect();
            if abs(pose.pitch) > pi/12 || abs(pose.yaw) > pi/8
                edgecolor = 'red';
            else
                edgecolor = 'blue';
            end

            set(hImage, 'cdata', data);
            set(hRect, 'Position', rect, 'EdgeColor', edgecolor);
            set(hShape, 'XData', shape(:,1), 'YData', shape(:,2));
            plotAxis(pose, 75, 75);
            drawnow;

            writeVideo(aviobj, getframe(gca));
       
        end
    end
end