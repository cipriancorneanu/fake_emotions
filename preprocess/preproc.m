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
expandRoi = [-50 -50 100 100];
roi = [1 1 vidObj.Width vidObj.Height];
offset = [0 0 0 0];
visualize = 1;
k = 0

%% Processing Loop
while hasFrame(vidObj)
    fprintf('Processing frame %d\n', k);
    
    % Read frame
    frame = readFrame(vidObj);    
      
    % Extract ROI   
    target = frame(roi(2):roi(2)+roi(4)-1, roi(1):roi(1)+roi(3)-1, :); 
    offset_target = [roi(1)-1 roi(2)-1];
    
    % Detect face     
    detection = round(step(faceDetector, target));    
       
    % If face detected 
    if ~isempty(detection)   
                
        % Fit shape
        %[shape, ~] = fitFrame(target, detection, model);

        % Place shape in frame
        %circles = [int16(repmat(offset_target,length(shape),1) + shape) ...
        %                 4*ones(length(shape),1)];
        rectangle = int16(detection + [offset_target 0 0]);
        
        % Paint on frame
        %frame = step(circleInserter, frame, circles);
        frame = step(rectInserter, frame, rectangle);
        
        % Define roi for next frame  
        roi = [offset_target 0 0] + detection + expandRoi;
        %offset = [roi(2) roi(1) 0 0]; 
    else
        % If face not detected reinit ROI and offset
       roi = [1 1 vidWidth vidHeight];
    end
    
    if visualize %&& rem(k,10)==0
        imshow(frame);pause(0.25);
    end
    
    %writeVideo(vidObjOut,frame);
           
    k = k+1;
end