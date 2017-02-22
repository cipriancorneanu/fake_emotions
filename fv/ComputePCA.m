function [reducedfeats, mean_col,U,S,V]  = ComputePCA(feats,dim)

%feat = cell2mat(feat);


mean_col = mean(feats);

data_cent = feats-repmat(mean_col,size(feats,1),1);

covData = data_cent'*data_cent;

[U S V] = svd(covData);


reducedfeats = feats*U(:,1:dim);

