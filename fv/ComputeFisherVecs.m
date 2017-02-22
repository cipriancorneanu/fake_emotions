clc;
clear all;
close all;

fid_lst = fopen('Cohn-Kanade-Traj-List.txt','rt');
[status results] = system(['wc -l Cohn-Kanade-Traj-List.txt']);
[numLines dummy]=sscanf(results,'%d %c');
numLines = numLines(1,1);
count=1;

load CohnKanadeTrajTrainFeatsSubSampPCAParamMBH.mat;

addpath(genpath('~/softs/vlfeat-0.9.20'));
vl_setup;
curr_dir = pwd;
maxNumIterations = 200;
pcadim = 64;
load ([curr_dir,'/CohnKanadeTrajFeatGMMs/TrainTrajMBHPCA64GMMsCents64.mat']);

while(1)
    tline = fgetl(fid_lst);
    fprintf('Processing file %d of %d\n',count,numLines);
    if tline == -1
        break;
    end;
    filename = gunzip(tline,pwd);
    [fpath,fname,fext] = fileparts(tline);
    [feats,stats,points] = readtraj(char(filename));
    feats = feats(:,205:end);
    feats = (feats-repmat(mean_col,size(feats,1),1))*U(:,1:pcadim);
    Vec = vl_fisher(feats',mu,sigma,w,'Improved');
    loc = strfind(fname,'_');
    loc1 = strfind(fname,'.');
    actor = fname(1:loc-1);
    instant = fname(loc+1:loc1-1);
    keyboard;
    save(['/data/hupba2/Derived/Cohn-Kanade-FV/',actor,'_',instant,'TrajMBHPCA64GMMsCents64','.mat'],'Vec');
    delete(filename{1});
    count = count +1;
end;
    