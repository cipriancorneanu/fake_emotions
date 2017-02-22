function [feat,trajstats,trajpoints] = readtrajrand(filename,percent)

%[status results] = system(['wc -l ',filename]);
%[numLines dummy]=sscanf(results,'%d %c');
%numLines = numLines(1,1);
%reduced = round(numLines*(percent/100));
%[status results] = system(['shuf -n ',num2str(reduced),' ',filename,' > output.txt']);
%fid = fopen('output.txt');
%count = 1;
%while(1)
%    tline = fgetl(fid);
%    if tline == -1
%       break;
%    end;
%    v = transpose(sscanf(tline,'%f'));
%    feat(count,:) = v(41:end);  
%    trajstats(count,:) = v(1);
%    trajpoints(count,:) = v(11:40);
%    count = count+1; 
%end;
fid = fopen(filename);
v = textscan(fid,'%f');
v = reshape(v{1},416,size(v{1},1)/416)';
%while(1)
%    tline = fgetl(fid);
%    if tline == -1
%       break;
%    end;
%    v = transpose(sscanf(tline,'%f'));
    Ind = randperm(size(v,1),round(5/100*size(v,1)));
    feat = v(Ind',21:end);
    trajstats = v(Ind',1:10);
    trajpoints = v(Ind',11:20);
%    count = count+1; 
%end;

fclose(fid);
