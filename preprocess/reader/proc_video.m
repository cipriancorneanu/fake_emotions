function [ ims, lms, emos ] = proc_video( path, fname )

    %% Construct a VideoReader object associated with the sample file.
    vidObj = VideoReader(strcat(path,fname));

    %% Define markers shape
    red = uint8([255 0 0]); green = uint8([0 255 0]);
    circleInserter = vision.ShapeInserter('Shape','Circles','Fill', 1, 'FillColor','Custom','CustomFillColor', green);
    rectInserter = vision.ShapeInserter('Shape','Rectangles', 'LineWidth', 2, 'BorderColor', 'Custom');

    %% Load face detector
    faceDetector = vision.CascadeObjectDetector('MinSize', [150 150],'MergeThreshold', 5); 

    %% Load geometry detector
    model = importdata('../models/model_sdm_cofw.mat');

    %% Init variables
    expansion = int16([-50 -50 100 100]);
    roi = int16([1 1 vidObj.Width vidObj.Height]);
    visualize = false;
    k = 1;
    emo_tokens = strsplit(fname, '.');
    numFrames = floor(vidObj.FrameRate*vidObj.Duration);
    ims = zeros(numFrames,200,200,3);
    lms = zeros(numFrames,29,2);
    emos = cell(numFrames,1);
    %down_sampling = 4;

    %% Extract Face Loop
    while k<2%hasFrame(vidObj) 
        fprintf('Processing frame %d\n', k);

        % Read frame
        frame = readFrame(vidObj);    

        % Extract ROI   
        target = frame(roi(2):roi(2)+roi(4)-1, roi(1):roi(1)+roi(3)-1, :); 
        %offset_target = int16([roi(1)-1 roi(2)-1]);

        % Detect face     
        detection = int16(round(step(faceDetector, target)));    

        % If face detected 
        if ~isempty(detection)   

            % Transform to frame coordinates
            detection = detection + [roi(1)-1 roi(2)-1 0 0];

            % Extract detection
            extraction = imresize(frame(detection(2):detection(2)+detection(4)-1,...
                                detection(1):detection(1)+detection(3)-1, :), [200,200]);

            % Fit shape
            [geom, ~] = fitFrame(extraction, [1 1 224 224], model);

            ims(k,:,:,:) = extraction;        
            lms(k,:,:,:) = geom;
            emos{k} = emo_tokens(1);

            % Define roi for next frame  
            roi = detection + expansion; 
        else
            % If face not detected reinit ROI and offset
           roi = [1 1 vidObj.Width vidObj.Height];
        end

        if visualize
            circles = [int16(geom) 4*ones(length(geom),1)];
            extraction = step(circleInserter, extraction, circles);
            imshow(extraction);pause(0.25);
        end
        k = k+1;
    end

end

