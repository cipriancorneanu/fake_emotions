clc;
clear all;
close all;

addpath(genpath('../../vlfeat-0.9.20'));
vl_setup;

addpath('/home/kulkarni/softs/libsvm-3.22/matlab');

trajPath = '~/Research/data/fake_emotions/improved_dense_trajectories/fake_faces/';
corr_recog = 0;
total_eg = 0;
confusion = zeros(12,12);   
    
[hog, hof, mbh, slices, lbls] = load_idt('/Users/cipriancorneanu/Research/data/fake_emotions/improved_dense_trajectories/');
descriptor = {hog, hof, mbh};

for d = 1:length(descriptor)
    fprintf('FV with descriptor %d\n', d);
    descr = descriptor{d};    
    folds = 1:length(descr);
    
    for i = 1:length(descriptor)   
        %% Define train and test
        folds_test = i;
        folds_train = setdiff(folds, folds_test);
        
        % Train
        for f = 1:length(folds_train)
            train{f} = descr{folds_train(f)};
            slices_train{f} = slices{folds_train(f)};
            labels_train{f} = lbls{folds_train(f)};
        end
        
        % Test
        test = descr{folds_test};
        slices_test = slices{folds_test};
        labels_test = lbls{folds_test};

        for numCenters = [64]
            %% Compute GMMs for from training data
            fprintf('Computing GMMs fold %d centers %d\n',i,numCenters);
            train_data = cat(1,train{:});
            sampled_train_data = train_data(randsample(size(train_data, 1), size(train_data, 1)/10), :);
            [mu,sigma,w] = ComputeKmeans(sampled_train_data,numCenters);
                 
            %% Compute FV per sequences train
            fishervecstrain = []; lbltrain=[]; fishervecstest=[];lbltest=[];
            for s = 1:length(slices_train)
                person_slice = slices_train{s};
                
                for seq = 1:size(person_slice,1)
                    % Slice sequence
                    sequence = train{s}(person_slice(seq,1):person_slice(seq,2), :);
                    size(sequence)
                    Vec = vl_fisher(sequence',mu,sigma,w,'Improved');
                    fishervecstrain = [fishervecstrain,Vec];
                    lbltrain = [lbltrain; labels_train{s}(seq)];
                end
            end
            
            %% Compute FV per sequences test
            person_slice = slices_test;
            for seq = 1:size(person_slice,1)
                % Slice sequence
                sequence = test(person_slice(seq,1):person_slice(seq,2), :) ;
                Vec = vl_fisher(sequence',mu,sigma,w,'Improved');
                fishervecstest = [fishervecstest,Vec];
                lbltest = [lbltest; labels_test(seq)];
            end        

            %{
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
            %}

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
end;
