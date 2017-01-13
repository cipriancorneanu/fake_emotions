addpath(genpath('../bosphorus')); savepath;

% Read data
bosphorus_path = '/Users/cipriancorneanu/Downloads/BosphorusDB/';
bReader = BosphorusReader(bosphorus_path);

% Read data
[rgb_dt, rgb_lm_dt, depth_dt, depth_lm_dt] = bReader.Read();

%{
file = 'bs000_YR_R90_0';

rgb = imread(strcat(file,'.png'));
rgb_lm = read_lm2file(strcat(file, '.lm2'));
depth = read_bntfile(strcat(file,'.bnt'));
depth_lm = read_lm3file(strcat(file, '.lm3'));
%}

%% Plot results
% Define subplot size
subplot_h = ceil(sqrt(numel(rgb_dt)));
%subplot_w = 2*subplot_w;

% Number of samples to display
N = 5;


for sample = 1:N%subplot_h%numel(rgb_dt)
    
    rgb = rgb_dt{sample};% imread(strcat(file,'.png'));
    rgb_lm = rgb_lm_dt{sample};% read_lm2file(strcat(file, '.lm2'));
    depth = depth_dt{sample};
    depth_lm = depth_lm_dt{sample};
    
    %% Mark landmarks on 2D image
    rgb_marked = mark_rgb(rgb, rgb_lm);
    
    size(rgb_lm);

    %% 3D
    [w h c] = size(rgb);

    depth_map = zeros(w, h);
    rgb_crd = depth(:,4:5);
    depth_value = depth(:,3);

    % Put all out of range values to default
    depth_value (depth_value < 0) = 1;

    % Coordinates are between 0 and 1. Multiply to get pixel coordinates.
    rgb_crd(:,1) = int16(rgb_crd(:,1)*w);
    rgb_crd(:,2) = int16(rgb_crd(:,2)*h);

    d = zeros(size(rgb,1),size(rgb,2));
    for i = 1:size(rgb_crd,1)
        d(rgb_crd(i,1),rgb_crd(i,2)) = depth_value(i);  
    end

    % 3D has lower resolution than 2D
    % Check if columns and rows are all zero and remove
    row_rm = ~any(d,2);
    col_rm = ~any(d);

    d(row_rm,:) = [];
    d(:,col_rm) = [];
    
    %Plot rgb
    subplot(2, N, sample);
    imshow(rgb_marked);

    % 3D colormap plot
    subplot(2, N, sample + N);
    imagesc(d');
    axis image;
    colormap hot;
    
end




