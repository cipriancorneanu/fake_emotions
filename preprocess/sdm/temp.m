function temp()
    colors = {'red', 'green', 'blue', 'cyan', 'magenta'};
    colors = { ...
        [0.09 0.23 0.68]*0.00 + [0.68 0.23 0.09]*1.00, ...
        [0.09 0.23 0.68]*0.25 + [0.68 0.23 0.09]*0.75, ...
        [0.09 0.23 0.68]*0.50 + [0.68 0.23 0.09]*0.50, ...
        [0.09 0.23 0.68]*0.75 + [0.68 0.23 0.09]*0.25, ...
        [0.09 0.23 0.68]*1.00 + [0.68 0.23 0.09]*0.00 ...
    };

    % Load libraries
    addpath(genpath('./libs'))
    
    % Load model
    model = load('temp_model.mat');
    model = model.model;
    
    % Load dataset
    data = load('data/lfpw_relabeled.mat');
    data = data.dataset;
    
    function [shape] = shapeFromPose(poseParams)
        dsp = poseParams(model.targets.idxsLocat);
        rot = poseParams(model.targets.idxsRotat);
        sca = poseParams(model.targets.idxsScale);
        
        tfm = [sca(1)*cos(rot) -sca(1)*sin(rot) 0 ; sca(2)*sin(rot) sca(2)*cos(rot) 0 ; 0 0 1];
        tfm(1:2,3) = [tfm(1,1)*dsp(1)+tfm(1,2)*dsp(2) tfm(2,1)*dsp(1)+tfm(2,2)*dsp(2)];
        tfm = pinv(tfm);
        
        % Get shape from pose parameters
        shape = bsxfun(@plus, reshape(model.pca.mean + poseParams(model.targets.idxsShape) * model.pca.transform, 2, [])' * tfm(1:2,1:2)', tfm(1:2,3)');
    end
    
    % Generate initializations
    model.initPoses = repmat(model.initPoses(1,:), [5 1]);
    model.initPoses(:,model.targets.idxsRotat(1)) = [-pi/6 -pi/12 0 +pi/12 +pi/6];
    model.initPoses(:,model.targets.idxsShape) = 0;
    
    function [hdl] = initFaceShape(color)
        hdl = zeros(1,8);
        hdl(1)  = plot([0 0 0 0 0], [0 0 0 0 0], 'Color', color);
        hdl(2)  = plot([0 0 0 0 0], [0 0 0 0 0], 'Color', color);
        hdl(3)  = plot([0 0 0 0 0], [0 0 0 0 0], 'Color', color);
        hdl(4)  = plot([0 0 0 0 0], [0 0 0 0 0], 'Color', color);
        hdl(5)  = plot([0 0 0 0],   [0 0 0 0],   'Color', color);
        hdl(6)  = plot([0 0 0 0],   [0 0 0 0],   'Color', color);
        hdl(7)  = plot([0 0 0 0 0], [0 0 0 0 0], 'Color', color);
        hdl(8)  = plot([0 0 0 0 0], [0 0 0 0 0], 'Color', color);
        hdl(9)  = plot([0 0],       [0 0],       'Color', color);
        hdl(10) = plot([0 0],       [0 0],       'Color', color);
        hdl(11) = plot([0 0],       [0 0],       '.', 'Color', color);
    end
    
    function plotFaceShape(hdl, shape)
        set(hdl(1),  'XData', shape([1 5 3 6 1],1),      'YData', shape([1 5 3 6 1],2));
        set(hdl(2),  'XData', shape([2 7 4 8 2],1),      'YData', shape([2 7 4 8 2],2));
        set(hdl(3),  'XData', shape([9 13 11 14 9],1),   'YData', shape([9 13 11 14 9],2));
        set(hdl(4),  'XData', shape([10 15 12 16 10],1), 'YData', shape([10 15 12 16 10],2));
        set(hdl(5),  'XData', shape([19 21 22 19],1),    'YData', shape([19 21 22 19],2));
        set(hdl(6),  'XData', shape([20 21 22 20],1),    'YData', shape([20 21 22 20],2));
        set(hdl(7),  'XData', shape([23 25 24 26 23],1), 'YData', shape([23 25 24 26 23],2));
        set(hdl(8),  'XData', shape([23 27 24 28 23],1), 'YData', shape([23 27 24 28 23],2));
        set(hdl(9),  'XData', shape([25 26],1),          'YData', shape([25 26],2));
        set(hdl(10), 'XData', shape([27 28],1),          'YData', shape([27 28],2));
        set(hdl(11), 'XData', shape([17 18],1),          'YData', shape([17 18],2))
    end

    % Prepare figure
    hFig = figure('Color', [1 1 1]);
    set(gca,'YDir','reverse', 'xtick', [], 'ytick', []);
    xlim([33 165]);
    ylim([18 140]);
    axis off;
    hold on;
    hdls = {};
    for i = [1 2 4 5 3], hdls{i} = initFaceShape(colors{i}); end
    hold off;

    % Prepare image counter
    ctr = 0;
    
    % Plot initializations
    for i = [1 2 4 5 3]
        shape = shapeFromPose(model.initPoses(3,:));
        plotFaceShape(hdls{i}, shape);
    end
    drawnow;
    imwrite(frame2im(getframe(gca)), ['centroid_' num2str(ctr) '.png']); ctr = ctr+1;
    
    nStep = 50;
    dpose = bsxfun(@minus, model.initPoses, model.initPoses(3,:)) / nStep;
    for iS = 1:nStep
        pause(0.030);
        for i = 1:5
            shape = shapeFromPose(model.initPoses(3,:) + dpose(i,:)*iS);
            plotFaceShape(hdls{i}, shape);
        end
        drawnow;
        imwrite(frame2im(getframe(gca)), ['centroid_' num2str(ctr) '.png']); ctr = ctr+1;
    end
    
    % Adjust to image
    [shape pose] = algorithmSDMTest_pose_alt(model, {data(5).face}, 'showOutputs', 0);
    pose(:,model.targets.idxsLocat) = model.initPoses(:,model.targets.idxsLocat);
    
    nStep = 50;
    dpose = (pose - model.initPoses) / nStep;
    for iS = 1:nStep
        pause(0.030);
        for i = 1:5
            shape = shapeFromPose(model.initPoses(i,:) + dpose(i,:)*iS);
            plotFaceShape(hdls{i}, shape);
        end
        drawnow;
        imwrite(frame2im(getframe(gca)), ['centroid_' num2str(ctr) '.png']); ctr = ctr+1;
    end
    %pause(5);
    
    nStep = 20;
    for i = 1:nStep
        thdls = [hdls{[1 2 4 5]}];
        for hdl = thdls
            pause(0.005);
            set(hdl, 'Color', min(get(hdl, 'Color')*1.2, 1));
        end
        drawnow;
        imwrite(frame2im(getframe(gca)), ['centroid_' num2str(ctr) '.png']); ctr = ctr+1;
    end
    
end

%     landmarks = [
%         0001  ... % (17) left_eye_pupil
%         0001  ... % (18) right_eye_pupil
%         10287 ... % (23) left_mouth_out	
%         6200  ... % (24) right_mouth_out
%         8807  ... % (25) mouth_center_top_lip_top
%         3281  ... % (26) mouth_center_top_lip_bottom	
%         6272  ... % (27) mouth_center_bottom_lip_top
%         6066  ... % (28) mouth_center_bottom_lip_bottom	
%         6704  ... % (29) chin
%     ];