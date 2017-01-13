function histogram = spatial_LBP_histogram(LBP_face, k, his_len)

%% This function calculates the spatial LBP histogram of the LBP image
%% devided into (k x k) regions.

dH = (size(LBP_face, 1)/k);
dL = (size(LBP_face, 2)/k);

histogram = zeros(1, his_len*(k^2));

kk = 0;
for ii = 1:k
    for jj = 1:k

        kk = kk + 1;
        sh = floor((ii-1)*dH + 1);
        eh = floor(ii*dH);
        sl = floor((jj-1)*dL + 1);
        el = floor(jj*dL);
        region = LBP_face(sh:eh, sl:el);

        Labels_vec = reshape(region, numel(region),1);

        % hist_Labels = hist(Labels_vec, his_len);
        hist_Labels = histc(Labels_vec, (0:his_len - 1));

        hist_Labels = hist_Labels/sum(hist_Labels);

        histogram(his_len*(kk-1)+1:his_len*kk) = hist_Labels;

    end;
end;

end