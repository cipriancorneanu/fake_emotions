function display_rgb( images, landmarks )
%DISPLAY_RGB Summary of this function goes here
%   Detailed explanation goes here

    % Iterate trough images and display with landmarks marked
    for i = 1:numel(images)   
        % Mark image
        m_img = mark_rgb(images{i}, landmarks{i});
        %m_img = Mark(obj, images{i}, landmarks{i}, emo, facs, context);

        imshow(m_img);

        pause(0.3);
    end
end 


