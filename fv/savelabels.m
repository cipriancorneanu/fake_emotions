clc;
clear all;
close all;

fid_lst = fopen('Cohn-Kanade-Traj-List.txt','rt');
[status results] = system(['wc -l Cohn-Kanade-Traj-List.txt']);
[numLines dummy]=sscanf(results,'%d %c');
numLines = numLines(1,1);
count=1;

actors = {};
instances = {};

filepath = '/data/hupba2/Datasets/Cohn-Kanade/Emotion/';
labels = [];
count = 1;
while(1)
tline = fgetl(fid_lst);
    fprintf('Processing file %d of %d\n',count,numLines);
    if tline == -1
        break;
    end;
    [fpath,fname,fext] = fileparts(tline); 
    loc = strfind(fname,'_');
    loc1 = strfind(fname,'.');
    actor = fname(1:loc-1);
    instant = fname(loc+1:loc1-1);
    
    emtfile = dir([filepath,actor,'/',instant,'/']);
    for i = 3:size(emtfile,1)
        labels(count) = load([filepath,actor,'/',instant,'/',emtfile(i).name]) ;
        actors{count} = actor;
        instances{count} = instant;
        count = count +1;
    end;

end;
fclose(fid_lst);

