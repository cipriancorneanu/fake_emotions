clc;
clear all;
close all;

actors = {'Laur', 'Alena', 'Ahmed', 'Andreas', 'Anton', 'Chris', 'Dona', 'Darwin', ...
    'Elmar', 'Francisca', 'Hassan', 'Iiris', 'Ivan', 'Kaisa', 'KarlGregori', 'Kirill', 'Laura', ...
    'LauraJogede', 'Mari-liis', 'Lucas', 'AlexanderMakarov', 'Aleksander', 'Mate', ...
    'Merilin', 'Nikita', 'Nina', 'Pavel', 'Pejman', ...
    'Remo', 'Richard', 'Suman', 'Roxanne', 'Reka', 'Zemaio', 'Vladimir', 'Vladimiz', 'Chris',...
    'nana', 'sinle', 'yiiri', 'Victor', ...
    'age', 'anne', 'Teddy', 'Asif', 'Rezwan', 'Sameer', 'Reena', 'toomas', 'Lembit', 'Yeh',...
    'Umesh', 'Airiin'};
actors_left = {'Kaisa', 'anee', 'Zemiao', 'AleksanderMakarov', 'Lukas'};
sort(actors)
emos = {'N2Sur', 'N2S', 'N2H', 'N2D', 'N2C', 'N2A', 'S2N2H', ....
            'H2N2D', 'H2N2C', 'H2N2A', 'D2N2Sur', 'H2N2S'};
labels = [3, 1, 0, 4, 2, 5, 6, 10, 8, 11, 9, 7];
pcadim = 64;
trajPath = '/data/hupba2/Derived/improved_dense_trajectories/fake_faces/';
path = '/home/corneanu/data/fake_emotions/idt/';

for i = 1 : length(actors)
    % sample the training folds
    TrainfeatsMBH = [];
    TrainfeatsHOG = [];
    TrainfeatsHOF = [];
    Trainstats = [];
    Trainpoints = [];
    lbl = [];
    slices = [];
    start = 1;
    fprintf('Accumulating trajectories for actor %d\n',i)
    act = actors{i};
    
    for l = 1:size(emos,2)
        fprintf('\tEmotion %d\n', l)      
        emo = emos{l};
        fname = [trajPath, emo, act, '.features.gz'];
        
        if exist(fname, 'file')
            lbl = [lbl; labels(l)];
            filename = gunzip(fname, pwd);

            [feats,stats,points] = readtraj(filename{1});
            TrainfeatsMBH = [TrainfeatsMBH;feats(:,205:end)];

            TrainfeatsHOG = [TrainfeatsHOG;feats(:,1:96)];
            TrainfeatsHOF = [TrainfeatsHOF;feats(:,97:204)];

            Trainstats = [Trainstats;stats];
            Trainpoints = [Trainpoints;points];

            slices = [slices; start, start+size(feats,1)];
            start = start + size(feats,1) + 1;
            delete(filename{1});
        end
    end
    if size(TrainfeatsMBH,1)>0
        [TrainfeatsMBH,~,~,~,~] = ComputePCA(TrainfeatsMBH,64);
        [TrainfeatsHOG,~,~,~,~] = ComputePCA(TrainfeatsHOG,32);
        [TrainfeatsHOF,~,~,~,~] = ComputePCA(TrainfeatsHOF,32);

        TrainfeatsMBH = single(TrainfeatsMBH);
        TrainfeatsHOG = single(TrainfeatsHOG);
        TrainfeatsHOF = single(TrainfeatsHOF);

        save(strcat(path, str(i), '.mat'), 'Trainstats', 'Trainpoints',...
        'TrainfeatsHOG', 'TrainfestsHOF', 'TrainfeatsMBH' , 'slices', 'lbl')
    else
       fprintf(strcat('Actor', act, ' not found \n')); 
    end
end
