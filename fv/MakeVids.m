clc;
clear all;
close all;



filepath = '/data/hupba2/Datasets/Cohn-Kanade/cohn-kanade-images';
Acts = dir(filepath);
siftpath = '/data/hupba2/Derived/Cohn-Kanade-Vids-landmarks/';
landmarkspath = '/data/hupba2/Datasets/Cohn-Kanade/Landmarks';

for i = 3:size(Acts,1)
    if Acts(i).isdir
       fileinst = ['/data/hupba2/Datasets/Cohn-Kanade/cohn-kanade-images/',Acts(i).name];
        instances = dir(fileinst);
        
        for j = 3:size(instances,1)
            if instances(j).isdir
               fprintf('doing act %s and instance %s \n',Acts(i).name,instances(j).name);
               pngpath = [fileinst,'/',instances(j).name];
               pngfiles = dir(pngpath);
               vidObj = VideoWriter([siftpath,Acts(i).name,'_',instances(j).name,'.avi'],'Uncompressed AVI');
               open(vidObj);
               for k = 3:size(pngfiles,1)
                   imgfile = [pngpath,'/',pngfiles(k).name];
                   [pathstr,filename,ext] = fileparts(imgfile);
                   if strcmp(ext,'.png')
                      Img = imread(imgfile);
                      landmarks = load([landmarkspath,'/',Acts(i).name,'/',instances(j).name,'/',filename,'_landmarks.txt']);
                      imshow(Img);
                      hold on;
                      plot(landmarks(:,1),landmarks(:,2),'+');
                      hold off;
                      frame = getframe;
                      frame.cdata = flipud(frame.cdata);
                      writeVideo(vidObj,flipud(frame));
                   end;
               end;
               close(vidObj);
            end;
        end;
    end;
end;