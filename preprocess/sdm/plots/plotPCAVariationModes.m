function [] = plotPCAVariationModes(mface, eigvec, eigval)
    nD = 2 + mod(numel(mface), 2);
    mShape = reshape(mface, nD, [])';
    nL = size(mShape, 1);
    
    NBASES = 3;
    
    %figure;
    %scatter(mShape(:,1), mShape(:,2));
    
    eVec = zeros(nL, nD, NBASES);
    eVal = zeros(NBASES,1);
    for i = 1:NBASES
        eVec(:,:,i) = reshape(eigvec(:,i), nD, [])';
        eVal(i) = sqrt(eigval(i));
    end
    
    stdvs = [-4 -2 2 4];
    figure;
    for i = 1:NBASES
        for j = 1:4
            tShape = mShape + eVec(:,:,i) * (eVal(i)*stdvs(j));
            
            subplot(NBASES, 4, (i-1)*4+j);
            set(gca,'YDir','reverse', 'xtick', [], 'ytick', []);
            axis([-100 100 -60 120]);
            hold on;
            
            scatter(tShape(:,1), tShape(:,2), 'fill', 'MarkerFaceColor', [1 0 0]);
            
            line(tShape(1:3,1), tShape(1:3, 2), 'Color', [0 0 0], 'LineWidth', 1.5);
            line(tShape(4:6,1), tShape(4:6, 2), 'Color', [0 0 0], 'LineWidth', 1.5);
            line(tShape(7:9,1), tShape(7:9, 2), 'Color', [0 0 0], 'LineWidth', 1.5);
            line(tShape(10:12,1), tShape(10:12, 2), 'Color', [0 0 0], 'LineWidth', 1.5);
            line(tShape(14:16,1), tShape(14:16, 2), 'Color', [0 0 0], 'LineWidth', 1.5);
            line(tShape([13 21 17],1), tShape([13 21 17], 2), 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
            line(tShape(18:20,1), tShape(18:20, 2), 'Color', [1 0 0], 'LineWidth', 1.8);
            
            hold off;
        end
    end
end