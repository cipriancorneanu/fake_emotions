function plotSampleFits(images, groundtruth, rdata)
    function plot2dfit(image, groundtruth, landmarks1, landmarks2)
        figure;
        imshow([image ; image]);
        hold on;
        scatter(groundtruth(:,1), groundtruth(:,2), 30, 'red');
        scatter(landmarks1(:,1), landmarks1(:,2), 30, 'cyan');
        
        groundtruth(:,2) = groundtruth(:,2) + size(image,1);
        landmarks2(:,2) = landmarks2(:,2) + size(image,1);
        
        scatter(groundtruth(:,1), groundtruth(:,2), 30, 'red');
        scatter(landmarks2(:,1), landmarks2(:,2), 30, 'cyan');
        hold off;
    end

    function plot3dfit(image, groundtruth, landmarks1, landmarks2, pose1, pose2)
        function plotAxis(pose, offx, offy)
            roll    = [ cos(pose.roll) sin(pose.roll) 0 ; -sin(pose.roll) cos(pose.roll) 0  ; 0 0 1                                 ];
            pitch   = [ 1 0 0                           ; 0 cos(pose.pitch) sin(pose.pitch) ; 0 -sin(pose.pitch) cos(pose.pitch)    ];
            yaw     = [ cos(pose.yaw) 0 sin(pose.yaw)   ; 0 1 0                             ; -sin(pose.yaw) 0 cos(pose.yaw)        ];
            rmat = yaw * pitch * roll;
            
            p1 = rmat * [30 0 0]';
            p2 = rmat * [0 30 0]';
            p3 = rmat * [0 0 30]';
            
            line([offx offx+p1(1)], [offy offy+(p1(2))], 'Color', 'm', 'LineWidth', 2);
            line([offx offx+p2(1)], [offy offy+(p2(2))], 'Color', 'c', 'LineWidth', 2);
            line([offx offx+p3(1)], [offy offy+(p3(2))], 'Color', 'r', 'LineWidth', 2);
        end
        
        figure;
        imshow([image ; image]);
        hold on;
        scatter(groundtruth(:,1), groundtruth(:,2), 30, 'red');
        scatter(landmarks1(:,1), landmarks1(:,2), 30, 'cyan');
        plotAxis(pose1, 20, 20);
        
        groundtruth(:,2) = groundtruth(:,2) + size(image,1);
        landmarks2(:,2) = landmarks2(:,2) + size(image,1);
        
        scatter(groundtruth(:,1), groundtruth(:,2), 30, 'red');
        scatter(landmarks2(:,1), landmarks2(:,2), 30, 'cyan');
        plotAxis(pose2, 20, 20+size(image,1));
        hold off;
    end

    for i = 500:520
        if size(groundtruth, 2) == 2
            plot2dfit(images{i}, groundtruth(:,:,i), rdata.sdm(1).shapes(:,:,i), rdata.parametric(1).shapes(:,:,i));
        else
            plot3dfit(images{i}, groundtruth(:,:,i), rdata.d2d(1).shapes(:,:,i), rdata.d3d(1).shapes(:,:,i), rdata.d2d(1).poses(i), rdata.d3d(1).poses(i));
        end
    end
end