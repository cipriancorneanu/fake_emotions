function [ img ] = mark_rgb( img, landmarks )
%MARK_LANDMARKS Mark landmarks on image and their corresponding ordering
%   Detailed explanation goes here

    %Define shape
    yellow = uint8([255 255 0]);
    shapeInserter = vision.ShapeInserter('Shape','Circles','Fill', 1, 'FillColor','Custom','CustomFillColor',yellow);

    %convert to color if black and white
    if size(img,3) == 1
        RGB = repmat(img,[1,1,3]); % convert I to an RGB image
    else
        RGB = img;
    end

    positions = [];
    values = 1:numel(landmarks);
    circles = [];

    for i = 1:size(landmarks,2) 
        position = [int16(landmarks(1,i)) int16(landmarks(2,i))]; %[x y]
        circle = [int16(landmarks(1,i)) int16(landmarks(2,i)) 5]; %[x y radius]

        positions = [positions; position];
        circles = [circles; circle];
    end

    values = 1:size(landmarks,2);
 
    %Insert points
    img = step(shapeInserter, RGB, circles);
    
    %Insert text
    img = insertText(img, positions, 'eee','FontSize',8,'AnchorPoint','LeftBottom');
end

