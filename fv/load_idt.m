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
    correction = [(0:11)',(1:12)'];
    
    hog = cell(1, length(files));
    hof = cell(1, length(files));
    mbh = cell(1, length(files));
    slices = cell(1, length(files));
    labels = cell(1, length(files));
    
    for i = 1:length(files)
       fname = strcat(path, files(i));
       data = load(fname{1});
       hog{i} = data.TrainfeatsHOF;
       hof{i} = data.TrainfeatsHOG;
       mbh{i} = data.TrainfeatsMBH;
       slices{i} = data.slices - correction;
       labels{i} = data.lbl;
    end
end

