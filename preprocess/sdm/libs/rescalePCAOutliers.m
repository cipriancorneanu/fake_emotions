function [data] = rescalePCAOutliers(data, eigenvalues)
    stdvs = sqrt(sum(bsxfun(@rdivide, data, sqrt(eigenvalues)') .^ 2, 2));
    idxsRescale = (stdvs > 3);
    if any(idxsRescale)
        facsRescale = rdivide(3, stdvs(idxsRescale));
        data(idxsRescale,:) = bsxfun(@times, data(idxsRescale,:), facsRescale);
    end
end