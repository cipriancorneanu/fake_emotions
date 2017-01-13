function [h] = plotFittedFace(image, landmarks, varargin)
    [nsL,nsD] = size(landmarks);
    
    h = imshow(image);
    hold on;
    
    % Show landmarks
    plot(landmarks(:,1), landmarks(:,2), 'o');
    
    if nsD == 3
        rotM = varargin{1}(1:3,1:3);
        
        % Show 3D landmarks
        scatter3(landmarks(:,1), landmarks(:,2), landmarks(:,3), 'o');
        
    elseif nsD == 2
        ASM3D = varargin{1};
        proj = varargin{2};
        w = varargin{3};
        rotM = proj.R;

        % Show 3D landmarks
        face3d = reshape(ASM3D.mean + w' * ASM3D.transform, 3, []);
        face3d(4,:) = 1;
        face3d = proj.transform * face3d;
        scatter3(face3d(1,:)',face3d(2,:)',face3d(3,:)');
    end
    
    % Show face orientation
    vert = [ ...
        0   50  0   0  ; ...
        0   0   50  0  ; ...
        0   0   0   50   ...
    ];

    vert = rotM * vert;
    vert = bsxfun(@plus, vert, [50 50 50]');
    line([vert(1,1),vert(1,2)], [vert(2,1),vert(2,2)], [vert(3,1),vert(3,2)], 'Color', 'y', 'LineWidth', 2);
    line([vert(1,1),vert(1,3)], [vert(2,1),vert(2,3)], [vert(3,1),vert(3,3)], 'Color', 'm', 'LineWidth', 2);
    line([vert(1,1),vert(1,4)], [vert(2,1),vert(2,4)], [vert(3,1),vert(3,4)], 'Color', 'c', 'LineWidth', 2);
    
    hold off;
end