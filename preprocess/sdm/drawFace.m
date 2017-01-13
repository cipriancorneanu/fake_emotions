function [] = drawFace( data, color, varargin )
%DRAWFACE Summary of this function goes here
%   Detailed explanation goes here
    
% Parameters
PrintShape = false;
nPars = length(varargin);
for i = 1:nPars        
    % Print shape
    if strcmpi(varargin{i}, 'PrintShape')
        PrintShape = true;
    end       
end
    
%Define geometry
left_brow = [1 2 3];
right_brow = [4 5 6];
nose = [11 13 14 15 12];
mouth = [16 17 18; 16 19 18; 16 20 18; 16 21 18];
left_eye = [7 8];
right_eye = [9 10];

bleye =  [ data(1,1) data(1,2)  data(2,1)  data(2,2);
           data(2,1) data(2,2)  data(3,1)  data(3,2);];
       
breye =  [ data(4,1) data(4,2)  data(5,1)  data(5,2);
           data(5,1) data(5,2)  data(6,1)  data(6,2);];
       
nose =   [ data(11,1) data(11,2)  data(13,1)  data(13,2);
           data(13,1) data(13,2)  data(14,1)  data(14,2);
           data(14,1) data(14,2)  data(15,1)  data(15,2);
           data(15,1) data(15,2)  data(12,1)  data(12,2);];
       
       
mouth =   [data(16,1) data(16,2)  data(17,1)  data(17,2);
           data(17,1) data(17,2)  data(18,1)  data(18,2);
           data(16,1) data(16,2)  data(19,1)  data(19,2);
           data(19,1) data(19,2)  data(18,1)  data(18,2);
           
           data(16,1) data(16,2)  data(20,1)  data(20,2);
           data(20,1) data(20,2)  data(18,1)  data(18,2);
          
           data(16,1) data(16,2)  data(21,1)  data(21,2);
           data(21,1) data(21,2)  data(18,1)  data(18,2);
           ];
       
leye = [ data(7,1) data(7,2)  data(8,1)  data(8,2) ];
reye = [ data(9,1) data(9,2)  data(10,1)  data(10,2)];

%Display
%figure();

%imshow(img); hold on; 
plot([mouth(:,1),mouth(:,3)],[mouth(:,2),mouth(:,4)],'Color', color,'LineWidth',2); hold on;
plot([nose(:,1),nose(:,3)],[nose(:,2),nose(:,4)],'Color',color,'LineWidth',2); hold on;
plot([leye(:,1),leye(:,3)],[leye(:,2),leye(:,4)],'Color',color,'LineWidth',2); hold on;
plot([reye(:,1),reye(:,3)],[reye(:,2),reye(:,4)],'Color',color,'LineWidth',2); hold on;
plot([bleye(:,1),bleye(:,3)],[bleye(:,2),bleye(:,4)],'Color',color,'LineWidth',2); hold on;
plot([breye(:,1),breye(:,3)],[breye(:,2),breye(:,4)],'Color',color,'LineWidth',2); hold on;

if PrintShape
    scatter3(data(:,1), data(:,2), data(:,3)+100);
end
  
end

