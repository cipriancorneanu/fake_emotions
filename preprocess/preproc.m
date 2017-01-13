%% Add path to dependencies
addpath(genpath('../../sota/rcpr'));

%% Construct a VideoReader object associated with the sample file.
path = '/Users/cipriancorneanu/Research/data/fake_emotions/';
vidObj = VideoReader(strcat(path,'N2S.MP4'));

%% Define markers shape
red = uint8([255 0 0]); green = uint8([0 255 0]);
circleInserter = vision.ShapeInserter('Shape','Circles','Fill', 1, 'FillColor','Custom','CustomFillColor', green);
rectInserter = vision.ShapeInserter('Shape','Rectangles', 'LineWidth', 2, 'BorderColor', 'Custom');

%% Load face detector
faceDetector = vision.CascadeObjectDetector('MinSize', [150 150],'MergeThreshold', 5); 

%% Init variables
expansion = int16([-50 -50 100 100]);
roi = int16([1 1 vidObj.Width vidObj.Height]);
visualize = 0;
k = 0;
        
%% Extract Face Loop
while hasFrame(vidObj) 
    if rem(k,4)==0
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

            extracted(:,:,:,k/4+1) = extraction;                

            % Define roi for next frame  
            roi = detection + expansion; 
        else
            % If face not detected reinit ROI and offset
           roi = [1 1 vidObj.Width vidObj.Height];
        end

        %{if visualize
        %    imshow(extraction);pause(0.25);
        %end
       
    end
    k = k+1;
end

%% Load geometry estimation model
model = importdata('./models/model_rcpr_300w.mat');

%% Align Face Loop
for i = 1:size(extracted,4)
   
    
end


%% Visualize
% Place shape in frame
%circles = [int16(repmat(offset_target,length(shape),1) + shape) ...
%                         4*ones(length(shape),1)];
%[shape, ~] = fitFrame(target, detection, model);

% Paint on frame
%frame = step(circleInserter, frame, circles);
%frame = step(rectInserter, frame, rectangle);