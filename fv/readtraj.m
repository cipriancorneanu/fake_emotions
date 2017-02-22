function [feat,trajstats,trajpoints] = readtraj(filename)

%[status results] = system(['wc -l ',filename]);
%[numLines dummy]=sscanf(results,'%d %c');
%numLines = numLines(1,1);
%reduced = round(numLines*(percent/100));
%[status results] = system(['shuf -n ',num2str(reduced),' ',filename,' > output.txt']);
%fid = fopen('output.txt');
%count = 1;
fid = fopen(filename);
v = textscan(fid,'%f');
v = reshape(v{1},436,size(v{1},1)/416)';
%while(1)
%    tline = fgetl(fid);
%    if tline == -1
%       break;
%    end;
%    v = transpose(sscanf(tline,'%f'));
    feat = v(:,21:end);  
    trajstats = v(:,1:10);
    trajpoints = v(:,11:20);
%    count = count+1; 
%end;

fclose(fid);