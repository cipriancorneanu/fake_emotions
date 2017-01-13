function [M, A_temp] = anchorPoints(M, anchors)
    % M: DxN matrix specifying the coordinates of a set of points
    % anchors: (1+D)xA matrix specifying the anchor indexs and the achor
    % coordinates
    
    if numel(anchors) == 0,
        A_temp = eye(size(M,1));
        return;
    end

    Xs = M(:,anchors(1,:));
    A_temp = anchors(2:end,:)*Xs'/(Xs*Xs');
    M = A_temp*M;
end

