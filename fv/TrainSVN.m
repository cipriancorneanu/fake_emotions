clc;
clear all;
close all;

addpath('/home/kulkarni/softs/libsvm-3.22/matlab');
load Cohn-Kanade-Labels.mat

fisherpath = '/data/hupba2/Derived/Cohn-Kanade-FV/';

feats = [];
for i =1:size(labels,2)
    temp = load([fisherpath,actors{i},'_',instances{i},'TrajMBHPCA64GMMsCents64.mat']);
    feats(i,:) = temp.Vec';
end;
faceModel = libsvmtrain(labels',feats,'-s 0 -t 0 -c 1000 -b 1 -v 10');



