clc;
clear all;
close all;

fid_lst = fopen('Cohn-Kanade-Traj-List.txt','rt');
[status results] = system(['wc -l Cohn-Kanade-Traj-List.txt']);
[numLines dummy]=sscanf(results,'%d %c');
numLines = numLines(1,1);
count=1;
Trainfeats = [];
Trainstats = [];
Trainpoints = [];
actor = {}
instant = {};
featCount = [];
count = 1;

while(1)
    tline = fgetl(fid_lst);
    fprintf('Processing file %d of %d\n',count,numLines);
    if tline == -1
        break;
    end;
    filename = gunzip(tline,pwd);
    [fpath,fname,fext] = fileparts(tline);
    [feats,stats,points] = readtrajrand(char(filename),5);
    Trainfeats = [Trainfeats;feats];
    Trainstats = [Trainstats;stats];
    Trainpoints = [Trainpoints;points];
    loc = strfind(fname,'_');
    loc1 = strfind(fname,'.');
    actor{count} = fname(1:loc-1);
    instant{count} = fname(loc+1:loc1-1);
    delete(filename{1});
    featCount(count) = size(feats,1); 
    count = count +1;
end;

fclose(fid_lst);

save -v7.3 CohnKanadeTrajTrainFeatsSubSamp.mat  Trainfeats Trainstats Trainpoints actor instant featCount;




% while(1)
%    tline = fgetl(fid_lst);
%    fprintf('Processing file %d of %d\n',count,numLines);
%    if tline == -1
%       break;
%    end 
%    loc = findstr(tline,'/');
%    actor = str2num(tline(loc(6)+2:loc(7)-1));
%    if ~((actor>=3)&(actor<=15))
%        filename = gunzip(tline,pwd);
%        [fpath,fname,fext] = fileparts(tline);
%        [feats,stats,points] = readtrajrand(char(filename),5);
%        Trainfeats = [Trainfeats;feats];
%        Trainstats = [Trainstats;stats];
%        Trainpoints = [Trainpoints;points];
%        delete(filename{1});
%    end;
%    count = count+1;
% end;