clc;
clear all;
close all;

addpath(genpath('../../vlfeat-0.9.20'));
vl_setup;

addpath('/home/kulkarni/softs/libsvm-3.22/matlab');

pcadim = 64;
subsamp = 5;
no_of_folds = 53;
actors = unique(actors);
actors  = sort(actors);
folds = round(linspace(1,size(actors,2)))

trajPath = '~/Research/data/fake_emotions/improved_dense_trajectories/fake_faces/';
corr_recog = 0;
total_eg = 0;
confusion = zeros(7,7);   
    
for i = 1:no_of_folds
    for numCenters = [64]
        fprintf('Computing GMMs fold %d centers %d\n',i,numCenters);
        [mu,sigma,w] = ComputeKmeans(Trainfeats,numCenters);
        fishervecstrain = [];
        lbltrain=[];
        fishervecstest=[];
        lbltest=[];
        for j = 1 : no_of_folds
            if i~=j
                for k = folds(j)+1:folds(j+1)
                    act = uniqAct{k};
                    ind = find(strcmp(actors,act));
                    for l = 1:size(ind,2)
                        filename = gunzip([trajPath,act,'_',instances{ind(l)},'.features.gz'],pwd);
                        [feats,stats,points] = readtraj(filename{1});
                        feats = feats(:,205:end);
                        feats = (feats-repmat(mean_col,size(feats,1),1))*U(:,1:pcadim);
                        Vec = vl_fisher(feats',mu,sigma,w,'Improved');
                        fishervecstrain = [fishervecstrain,Vec];
                        delete(filename{1});
                        lbltrain = [lbltrain,labels(ind(l))];
                    end;
                end;
            else
                for k = folds(j)+1:folds(j+1)
                    act = uniqAct{k};
                    ind = find(strcmp(actors,act));
                    for l = 1:size(ind,2)
                        filename = gunzip([trajPath,act,'_',instances{ind(l)},'.features.gz'],pwd);
                        [feats,stats,points] = readtraj(filename{1});
                        feats = feats(:,205:end);
                        feats = (feats-repmat(mean_col,size(feats,1),1))*U(:,1:pcadim);
                        Vec = vl_fisher(feats',mu,sigma,w,'Improved');
                        fishervecstest = [fishervecstest,Vec];
                        delete(filename{1});
                        lbltest = [lbltest,labels(ind(l))];
                    end;
                end;
            end;
        end;
        fprintf('Training SVMs fold %d centers %d\n',i,numCenters);
        faceModel = libsvmtrain(lbltrain',fishervecstrain','-s 0 -t 0 -c 100 -b 1 -q ');
        fprintf('Testing SVMs fold %d centers %d\n',i,numCenters);
        [predicted_label, accuracy,prob_estimates] = libsvmpredict(lbltest', fishervecstest', faceModel, '-b 1');
        %fprintf('accuracy=%f\n',accuracy);
        for j = 1:size(predicted_label,1)
            if predicted_label(j)== lbltest(j)
                corr_recog = corr_recog+1;
            end;
            total_eg = total_eg+1;
            confusion(lbltest(j),predicted_label(j)) = confusion(lbltest(j),predicted_label(j)) +1;
        end;
    end;
end;
