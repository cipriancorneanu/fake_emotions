
function [mu,sigma,w] = ComputeKmeans(feats,numCenters) 

maxNumIterations = 200;

[mu, sigma, w] = vl_gmm(feats', ...
    numCenters, ...
    'NumRepetitions',8, ...
    'MaxNumIterations', maxNumIterations);
