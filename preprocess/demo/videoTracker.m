visualize = true;

addpath(genpath('./sdm'))

%% Construct a VideoReader object associated with the sample file.
vidObj = VideoReader('../data/N2S.MP4');

%% Determine the height and width of the frames.
vidHeight = vidObj.Height;
vidWidth = vidObj.Width;

s = struct('cdata',zeros(vidHeight,vidWidth,3,'uint8'),...
    'colormap',[]);

%% Define shape
red = uint8([255 0 0]); green = uint8([0 255 0]);
circleInserter = vision.ShapeInserter('Shape','Circles','Fill', 1, 'FillColor','Custom','CustomFillColor', green);
rectInserter = vision.ShapeInserter('Shape','Rectangles', 'LineWidth', 2, 'BorderColor', 'Custom');

%% Prepare video output file.
vidObjOut = VideoWriter('../data/out2.avi');
open(vidObjOut);

%% Load landmark detector
model = importdata('../models/model_sdm_cofw.mat');

%% Load face detector
faceDetector = vision.CascadeObjectDetector('MinSize', [150 150],'MergeThreshold', 5); 

%% Init variables
k = 1;
expandRoi = [-50 -50 100 100];
%rect = round(step(faceDetector, s(k).cdata));
roi = [1 1 vidObj.Width vidObj.Height];
offset = [0 0 0 0];
    
%% Read one frame at a time using readFrame until the end of the file is reached.
%% Append data from each video frame to the structure array.        
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
        [shape, ~] = fitFrame(target, detection, model);

        % Place shape in frame
        circles = [int16(repmat(offset_target,length(shape),1) + shape) ...
                         4*ones(length(shape),1)];
        rectangle = int16(detection + [offset_target 0 0]);
        
        % Paint on frame
        frame = step(circleInserter, frame, circles);
        frame = step(rectInserter, frame, rectangle);
        
        % Define roi for next frame  
        roi = [offset_target 0 0] + detection + expandRoi;
        %offset = [roi(2) roi(1) 0 0]; 
    else
        % If face not detected reinit ROI and offset
       roi = [1 1 vidWidth vidHeight];
    end
    
    if visualize
        imshow(frame);pause(0.25);
    end
    
    writeVideo(vidObjOut,frame);
           
    k = k+1;
end

% Close the file.
close(vidObjOut);
    
%% Save/Display

%% Resize the current figure and axes based on the video's width and height.
%% Then, play the movie once at the video's frame rate using the movie function.
if visualize   
    set(gcf,'position',[150 150 vidObj.Width vidObj.Height]);
    set(gca,'units','pixels');
    set(gca,'position',[0 0 vidObj.Width vidObj.Height]);
    movie(s,1,vidObj.FrameRate);
    close
end



