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
            images = [];
            landmarks = [];
            emotions = {};
            
            for person = 1:length(PersonSubdirs) 
                fprintf(strcat('Reading person:\t', PersonSubdirs{person}, '\n'));                              
                Path = strcat(obj.DataPath, PersonSubdirs{person}, filesep);
                             
                % Get all video files from each subdir. each one corrsponds
                % to a different emotion
                rgb_files = GetFiles(obj, Path, '*.MP4');              

                for i = 1:length(rgb_files)
                    fprintf(strcat('Reading video:\t', rgb_files{i}, '\n'));  
                    if numel(rgb_files{i})>1
                        [ims, lms, emos] = proc_video(Path, rgb_files{i});
                    else
                        fprintf('\t\t No files found\n');
                    end
                    images = [images; ims];
                    landmarks = [landmarks; lms];
                    emotions = [emotions; emos];
                end
                
                % Write to file
                out = {'ims', 'lms', 'emos'; images, landmarks, emotions};
                save(strcat(obj.SavePath, num2str(person), '.mat'), 'out', '-v7.3');
            end
        end
    end  
end



