function [modl, newV] = align3D(p2d, p3d)
    %% remove translation
    p22 = mean(p2d, 2);
    p33 = mean(p3d, 2);    
    p2center = bsxfun(@minus, p2d, p22);
    p3center = bsxfun(@minus, p3d, p33);

    function [U,Q] = vgg_rq(S)
        S = S';
        [Q,U] = qr(S(end:-1:1,end:-1:1));
        Q = Q';
        Q = Q(end:-1:1,end:-1:1);
        U = U';
        U = U(end:-1:1,end:-1:1);

        if det(Q)<0
          U(:,1) = -U(:,1);
          Q(1,:) = -Q(1,:);
        end
    end
    
    % Find best rotation matrix (solve AR * p3c = p2c) and scaling (s)
    AR    = (p2center * p3center') / (p3center*p3center');
    A     = [AR ; cross(AR(1,:),AR(2,:))];
    [K,R] = vgg_rq(A(:,1:size(A,1)));
    Sgn   = diag(sign(diag(K)));
    K     = K * Sgn;
    R     = Sgn * R;
    s     = (K(1,1) + K(2,2))/ 2;

    % Compute displacement transforms
    disp3D = eye(4);    disp3D(1:3, 4) = -p33;
    disp2D = eye(4);    disp2D(1:2, 4) = p22;
    
    % Compute transform
    tfm = eye(4);
    tfm(1:3,1:3) = s * R;
    tfm = disp2D * tfm * disp3D;
    
    % Compute projection
    prj = tfm(1:3,:);
    prj(3,:) = [0 0 0 1];
    
    % Compute aligned 3D shape
    if nargout > 1
        p3d(4,:) = 1;
        newV = tfm * p3d;
        newV = newV(1:3,:);
    end
    
    % Store model
    modl = struct( ...
        'center2D', p22, ...
        'center3D', p33, ...
        'R', R, ...
        'scale', s, ...
        'transform',  tfm, ...
        'projection', prj ...
    );
end