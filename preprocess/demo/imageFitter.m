function [ ] = imageFitter( file_name )
%IMAGEFITTERR Summary of this function goes here
%   Detailed explanation goes here

visualize = true;

%% Define shape
red = uint8([255 0 0]); green = uint8([0 255 0]);
circleInserter = vision.ShapeInserter('Shape','Circles','Fill', 10, 'FillColor','Custom','CustomFillColor', green);
rectInserter = vision.ShapeInserter('Shape','Rectangles', 'LineWidth', 2, 'BorderColor', 'Custom');

%% Load landmark detector
%model = importdata('../train/results/model_sdm_cofw_cvc.mat');
model = importdata('../train/results/model_sdm_cofw.mat');

%% Load face detector
faceDetector = vision.CascadeObjectDetector('MinSize', [50 50],'MergeThreshold', 5); 

%% Init variables
k = 1;
expandRoi = [-50 -50 100 100];
contractRoi = [50 50 -100 -100];
offset = [0 0 0 0];
ipath = './faces/';
  
frame  = imread(strcat(ipath, file_name, '.jpg'));

[fHeight, fWidth, ~] = size(frame);
roi = [1 1 fWidth-1 fHeight-1];

% Extract ROI   
target = frame(roi(2):roi(2)+roi(4)-1, roi(1):roi(1)+roi(3)-1, :); 
offset_target = [roi(1)-1 roi(2)-1];

% Detect face     
detection = round(step(faceDetector, target));    
 
% If face not detected take whole frame minus borders 
if isempty(detection)   
    detection = roi + contractRoi;
end

% Fit shape
[shape, ~] = fitFrame(target, detection, model);

% Place shape in frame
circles = [int16(repmat(offset_target,length(shape),1) + shape) ...
                 5*ones(length(shape),1)];
rectangle = int16(detection + [offset_target 0 0]);

% Paint on frame
frame = step(circleInserter, frame, circles);

imshow(frame);

end

