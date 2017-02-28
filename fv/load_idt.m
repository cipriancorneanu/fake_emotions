function [hog, hof, mbh, slices, labels ] = load_idt( path )
%LOAD_IDT Summary of this function goes here
%   Detailed explanation goes here

    data = dir(path);
    
    % Find the index for directories      
    index = ~[data.isdir]; 

    % Get a list of the subject subdirectories
    files = {data(index).name};  

    % Remove current and parent directory from the list
    invalid_index = find(ismember({data(index).name},{'.','..', '.DS_Store'}));

    % Remove current and parent dir from subdirs
    files(invalid_index) = []
    
    len = length(files);
    
    hog = cell(1,len);
    hof = cell(1,len);
    mbh = cell(1,len);
    slices = cell(1,len);
    labels = cell(1,len);
    
    for i = 1:len
       fname = strcat(path, files(i));
       data = load(fname{1});
       hg = data.TrainfeatsHOG - repmat(min(data.TrainfeatsHOG), size(data.TrainfeatsHOG, 1), 1); 
       hf = data.TrainfeatsHOF - repmat(min(data.TrainfeatsHOF), size(data.TrainfeatsHOF, 1), 1); 
       mh = data.TrainfeatsMBH - repmat(min(data.TrainfeatsMBH), size(data.TrainfeatsMBH, 1), 1); 
       
       hog{i} = hg ./ repmat(max(hg), size(hg, 1), 1);
       hof{i} = hf ./ repmat(max(hf), size(hf, 1), 1);
       mbh{i} = mh ./ repmat(max(mh), size(mh, 1), 1);
       slices{i} = data.slices - [(0:size(data.slices,1)-1)',(1:size(data.slices,1))'];
       labels{i} = data.lbl;
    end
end

