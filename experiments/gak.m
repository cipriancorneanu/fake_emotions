clear
path = '/Users/cipriancorneanu/Research/data/fake_emotions/geoms/';
load(strcat(path, 'femo_geom_proc.mat'))

addpath(genpath('../libs'));

windowSize = 20;
b = (1/windowSize)*ones(1,windowSize);
a = 1;

%{
% Plot first and second parameter for 10 persons
figure(1)
param = 2; pers = 2;
for i = 1:6
    true = X{i+(pers-1)*6}(:,param);
    true = filter(b,a,true - repmat(mean(true),length(true),1));
    
    fake = X{i+(pers-1)*6+324}(:,param);
    fake = filter(b,a,fake - repmat(mean(fake),length(fake),1));
       
    true = (true - min(true))/ max(true-min(true));
    fake = (fake - min(fake))/ max(fake-min(fake));
    
norm      
%}

%% Norm and clean data
%{
for i = range(1, length(X))
    for j = range(1, size(X{i},2))
        
    end
end
%}

%% Compute hyperparmeters
N = length(X);
% Compute median length on all dataset
for i = 1:N
   lens(i) = length(X{i});
end
median_length = median(lens);

% Compute median difference norm on whole dataset
n = 20;
diff_norms = double([]);
i_range = floor(rand(1,n)*N);
j_range = floor(rand(1,n)*N);
for i = i_range
   for j = j_range
       if(length(X{i}) && length(X{j}))
           %fprintf('i=%d, j=%d\n', i, j);
           slice = 1:min(length(X{i}), length(X{j}));
           diff_norms = [diff_norms, norm(X{i}(slice,:) - X{j}(slice,:))]; 
       end
   end
end
median_diff_norm = median(diff_norms);

% Parameter grid search
for s = [0.2, 0.5, 1, 2, 5]
    for t = [0.2, 0.5]           
        T = int16(floor(t*median_length));
        S = s;%s*median_diff_norm*sqrt(median_length);        
        gak = exp(logGAK(X{1},X{2},S,0)-.5*(logGAK(X{1},X{1},S,0)+logGAK(X{2},X{2},S,T)));
        fprintf('T = %f, S = %f, gak = %f\n', T, S, gak);
    end
end

   
    %{
    % True
    subplot(6,2,2*i-1);
    plot(true);
    title(strcat('True', num2str(i)));
    
    % Fake
    subplot(6,2,2*i);
    plot(fake);
    title(strcat('Fake', num2str(i)));
end

figure(2)
for i = 1:6
    index1 = (i-1)*6+5
    emo1 = X{index1}(:,param);
    emo1 = filter(b,a,emo1 - repmat(mean(emo1),length(emo1),1));
    
    index2 = (i-1)*6+3
    emo2 = X{index2}(:,param);
    emo2 = filter(b,a,emo2 - repmat(mean(emo2),length(emo2),1));
    
    % True
    subplot(6,2,2*i-1);
    plot(emo1);
    title(strcat('Emo1', num2str(i)));
    
    subplot(6,2,2*i);
    plot(emo2);
    title(strcat('Emo2', num2str(i)));
end
%}

