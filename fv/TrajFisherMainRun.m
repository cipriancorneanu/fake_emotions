clc;
clear all;
close all;

addpath(genpath('../../vlfeat-0.9.20'));
vl_setup;

addpath('/home/kulkarni/softs/libsvm-3.22/matlab');
savepath('/home/corneanu/data/fake_emotions/idt/gmm/full/')

trajPath = '/home/corneanu/data/fake_emotions/idt/';
corr_recog = 0;
total_eg = 0;
confusion = zeros(12,12);   
boost = 20;
    
[hog, hof, mbh, slices, lbls] = load_idt(trajPath);
descriptor = {hog};

res = zeros(length(descriptor), length(descriptor{1}));

for d = 1:length(descriptor)
    fprintf('-------->FV with descriptor %d\n', d);
    descr = descriptor{d};    
    folds = 1:length(descr);
    
    for i = 1:length(descr)   
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

        numCenters = [256, 512];
        for n = 1:len(numCenters)
            %% Compute GMMs for from training data
            fprintf('Computing GMMs fold %d centers %d\n',i,numCenters(n));
            train_data = cat(1,train{:});
	    
            sampled_train_data = train_data(randsample(size(train_data, 1), int64(size(train_data, 1)/10)), :);

            [mu,sigma,w] = ComputeKmeans(sampled_train_data,numCenters(n));
            save([savepath,'gmm_', d, '_', i, '_', n, '.mat'], 'mu', 'sigma', 'w');
                 
            %% Compute FV per sequences train
            fishervecstrain = []; lbltrain=[]; fishervecstest=[];lbltest=[];
            for s = 1:length(slices_train)
                person_slice = slices_train{s};

                % Slice sequence
                for seq = 1:size(person_slice,1)
                    sequence = train{s}(person_slice(seq,1):person_slice(seq,2), :);
                    Vec = vl_fisher(sequence',mu,sigma,w,'Improved');
                    fishervecstrain = [fishervecstrain,Vec];
                    lbltrain = [lbltrain; labels_train{s}(seq)];                    
                end               
                
                %{
                for seq = 1:size(person_slice,1)
                len_slice = (person_slice(seq,2) - person_slice(seq,1));
                    len = int64(0.55*len_slice); 
                    starts = randsample(int64(0.4*len_slice), boost);

                    for st = 1:length(starts)
                        sequence = train{s}(person_slice(seq,1)+starts(st): ...
                        person_slice(seq,1)+starts(st)+len, :);
                        Vec = vl_fisher(sequence',mu,sigma,w,'Improved');
                        fishervecstrain = [fishervecstrain,Vec];
                        lbltrain = [lbltrain; labels_train{s}(seq)];
                    end
                end
                %}
            end
            
            %% Compute FV per sequences test
            person_slice = slices_test;
            for seq = 1:size(person_slice,1)
                % Slice sequence
                sequence = test(person_slice(seq,1):person_slice(seq,2), :);
                Vec = vl_fisher(sequence',mu,sigma,w,'Improved');
                fishervecstest = [fishervecstest,Vec];              
                lbltest = [lbltest; labels_test(seq)];
            end        

            %[fishervecstrain, mean,  U, S, V] = ComputePCA(fishervecstrain', 64);
            %fishervecstest = (fishervecstest'-repmat(mean, size(fishervecstest',1),1))*U(:,1:64);
            %size(fishervecstrain)
            %size(fishervecstest)

            fprintf('Training SVMs fold %d centers %d\n cost %d\n ',i,numCenters, c);
            options = '-s 0 -t 0 -c 100 -b 1 -q ';
            faceModel = libsvmtrain(double(lbltrain),double(fishervecstrain'), options);
            fprintf('Testing SVMs fold %d centers %d\n',i,numCenters);

            [predicted_label, accuracy,prob_estimates] = libsvmpredict(double(lbltest), double(fishervecstest'), faceModel, '-b 1');
            %fprintf('accuracy=%f\n',accuracy);
            
            for j = 1:size(predicted_label,1)
                if predicted_label(j)== lbltest(j)
                    corr_recog = corr_recog+1;
                end;
                
                total_eg = total_eg+1;
                confusion(lbltest(j)+1,predicted_label(j)+1) = confusion(lbltest(j)+1,predicted_label(j)+1) +1;
            end;
            confusion
            save([savepath,'conf_', num2str(d), '_', num2str(i), '_', num2str(n), '.mat'], 'confusion');
            res(d,i,n) = accuracy(1);  
        end; 
   end;
end;

res
