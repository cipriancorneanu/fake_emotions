function [results] = experimentPoseFitting(model2d, model3d, varargin)
    V_PITCH  = [-90 -75 -60 -45 -30 -15 0 +15 +30 +45 +60 +75 +90] * 0.75;
    V_YAW    = [-90 -75 -60 -45 -30 -15 0 +15 +30 +45 +60 +75 +90];
    npitch = length(V_PITCH);
    nyaw   = length(V_YAW);
    
    % Add 'libs' folder and subfolders to path
    addpath(genpath('libs'))
    
    % SET ALGORITHM PARAMETERS
    % ------------------------------------------------------------
    
    parsInit = {[] [] [] []};
    [shapes2d,poses2d,shapes3d,poses3d] = deal(parsInit{:});
    
    nPars = length(varargin);
    for i = 1:2:nPars
        pN = varargin{i};
        pV = varargin{i+1};
        
        % Number of cascade steps
        if strcmpi(pN, 'data2d')
            shapes2d = pV{1};
            poses2d  = pV{2};
            
        % Level of training data boosing
        elseif strcmpi(pN, 'data3d')
            shapes3d = pV{1};
            poses3d  = pV{2};
            
        end
    end
    
    % LOAD DATA & PREPARE RESULTS
    % ------------------------------------------------------------
    
    tdata = load('data/pointing04.mat');
    tdata = tdata.data;
    
    % Prepare return structure
    results = struct( ...
        'd2d', repmat(struct('series', [], 'shapes', [], 'poses', struct('roll', 0, 'pitch', 0, 'yaw', 0), 'errors_pitch', [], 'errors_yaw', []), [npitch, nyaw]), ...
        'd3d', repmat(struct('series', [], 'shapes', [], 'poses', struct('roll', 0, 'pitch', 0, 'yaw', 0), 'errors_pitch', [], 'errors_yaw', []), [npitch, nyaw])  ...
    );

    % PERFORM EXPERIMENT
    % ------------------------------------------------------------

    % Process images with 2D pose recovery
    if isempty(shapes2d) && isempty(poses2d)
        [shapes2d, poses2d, ~, ~] = algorithmSDMTest_pose2d(model2d, {tdata.face}, 'showOutputs', 0);
        for i = 1:length(poses2d)
            poses2d(i).roll  = poses2d(i).roll  * 360/(2*pi);
            poses2d(i).pitch = -poses2d(i).pitch * 360/(2*pi);
            poses2d(i).yaw   = -poses2d(i).yaw   * 360/(2*pi);
        end
    end
    
    % Process images with 3D pose recovery
    if isempty(shapes3d) && isempty(poses3d)
        [shapes3d, poses3d, ~, ~] = algorithmSDMTest_pose3d(model3d, {tdata.face}, 'showOutputs', 0);
        for i = 1:length(poses3d)
            poses3d(i).roll  = poses3d(i).roll  * 360/(2*pi);
            poses3d(i).pitch = -poses3d(i).pitch * 360/(2*pi);
            poses3d(i).yaw   = -poses3d(i).yaw   * 360/(2*pi);
        end
    end
    
    save('tmp_posedata.mat', 'shapes2d', 'poses2d', 'shapes3d', 'poses3d');
    
    % Process images and bin results according to pitch and yaw orientations
    for i = 1:length(tdata)
        % Get current image pitch and yaw indexs
        ixPitch = find(V_PITCH == tdata(i).pose3d.pitch);
        ixYaw = find(V_YAW == tdata(i).pose3d.yaw);
    
        % Append errors to the corresponding bins (2D)
        results.d2d(ixPitch, ixYaw).series(end+1) = tdata(i).person*10 + tdata(i).series;
        results.d2d(ixPitch, ixYaw).shapes(:,:,end+1) = shapes2d(i);
        results.d2d(ixPitch, ixYaw).poses(end+1) = poses2d(i);
        results.d2d(ixPitch, ixYaw).errors_pitch(end+1) = abs(tdata(i).pose3d.pitch - poses2d(i).pitch);
        results.d2d(ixPitch, ixYaw).errors_yaw(end+1) = abs(tdata(i).pose3d.yaw - poses2d(i).yaw);
        
         % Append errors to the corresponding bins (3D)
        results.d3d(ixPitch, ixYaw).series(end+1) = tdata(i).person*10 + tdata(i).series;
        results.d3d(ixPitch, ixYaw).shapes(:,:,end+1) = shapes3d(i);
        results.d3d(ixPitch, ixYaw).poses(end+1) = poses3d(i);
        results.d3d(ixPitch, ixYaw).errors_pitch(end+1) = abs(tdata(i).pose3d.pitch - poses3d(i).pitch);
        results.d3d(ixPitch, ixYaw).errors_yaw(end+1) = abs(tdata(i).pose3d.yaw - poses3d(i).yaw);
    end
    
    % Save results
    save('results/processed_posefit.mat', 'results');
end