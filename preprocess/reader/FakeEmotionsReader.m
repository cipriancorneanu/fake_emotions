classdef FakeEmotionsReader < DataReader
    properties
    end
    
    methods        
        %% Constructor
        function obj = FakeEmotionsReader(data_path, save_path)
            obj.DataPath = data_path;
            obj.SavePath = save_path;
        end
        
        function [] = Read(obj)         
            % Get all subdirectories in path; each one corresponds to a
            % different person
            PersonSubdirs = GetSubdirs(obj, obj.DataPath);
            
            %% Load geometry detector
            model = importdata('../models/model_sdm_cofw.mat');
            
            for person = 1:length(PersonSubdirs) 
                fprintf(strcat('Reading person:\t', PersonSubdirs{person}, '\n'));                              
                PathPers = strcat(obj.DataPath, PersonSubdirs{person}, filesep);
                images = [];
                landmarks = [];
                emotions = {};
                visualize = false;
            
                % Get all subdirs in path; each one corresponds to a
                % different label
                LabelSubdirs = GetSubdirs(obj, PathPers);
                
                for label = 1:length(LabelSubdirs)
                    fprintf(strcat('Reading emotion:\t', LabelSubdirs{label}, '\n'));  
                    
                    % Get all files from each subdir. Each one corresponds
                    % to a different frame
                    PathEmo = strcat(PathPers, LabelSubdirs{label}, filesep);
                    rgb_files = GetFiles(obj, PathEmo, '*.png');   
                    
                    % Sort files by taking into account fname format
                    if strcmp(rgb_files{1}(1),'f')
                        tokens = cellfun(@(x) strsplit(x,'.'), rgb_files, 'UniformOutput', false);
                        tokens = cellfun(@(x) x(1), tokens);
                        [tokens, idx] = sort(cellfun(@(x) str2num(x(6:end)), tokens));
                        rgb_files = rgb_files(:,idx);                        
                    else
                        tokens = cellfun(@(x) strsplit(x,'['), rgb_files, 'UniformOutput', false);
                        tokens = cellfun(@(x) x(1), tokens);
                        [tokens, idx] = sort(cellfun(@(x) str2num(x), tokens));
                        rgb_files = rgb_files(:,idx);
                    end
                                    
                    % Init containers
                    numFrames = length(rgb_files);
                    ims = zeros(numFrames,224,224,3);
                    lms = zeros(numFrames,29,2);
                    emos = cell(numFrames,1);

                    for i = 1:30%length(rgb_files)
                        fprintf(strcat('Reading emotion:\t', int2str(i), '\n'));
                        
                        frame = imread(strcat(PathEmo, rgb_files{i}));

                        % Fit shape
                        [geom, ~] = fitFrame(frame, [1 1 224 224], model);

                        ims(i,:,:,:) = frame;        
                        lms(i,:,:) = geom;
                        emos{i} = LabelSubdirs{label};
                        
                        if visualize
                           imshow(frame);
                           hold on;
                           scatter(geom(:,1), geom(:,2), 'r.');
                           pause(0.5);
                        end
                    end
                    
                    images = [images; ims];
                    landmarks = [landmarks; lms];
                    emotions = [emotions; emos];
                end
                
                % Write to file
                out = {images, landmarks, emotions};
                save(strcat(obj.SavePath, num2str(person), '.mat'), 'out', '-v7.3');
            end
        end
    end  
end



