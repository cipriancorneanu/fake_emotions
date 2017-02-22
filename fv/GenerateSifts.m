clc;
clear all;
close all;


addpath(genpath('/home/kulkarni/softs/vlfeat-0.9.20'));

filepath = '/data/hupba2/Datasets/Cohn-Kanade/cohn-kanade-images';
Acts = dir(filepath);
siftpath = '/data/hupba2/Derived/Cohn-Kanade-Vids/';

for i = 3:size(Acts,1)
    if Acts(i).isdir
       fileinst = ['/data/hupba2/Datasets/Cohn-Kanade/cohn-kanade-images/',Acts(i).name];
        instances = dir(fileinst);
        
        for j = 3:size(instances,1)
            if instances(j).isdir
               fprintf('doing act %s and instance %s \n',Acts(i).name,instances(j).name);
               pngpath = [fileinst,'/',instances(j).name];
               pngfiles = dir(pngpath);
               featsift = [];
               framecount = [];
               count = 1;
               for k = 3:size(pngfiles,1)
                   imgfile = [pngpath,'/',pngfiles(k).name];
                   [pathstr,filename,ext] = fileparts(imgfile);
                   if ~strcmp(ext,'.png')
                       continue;
                   end;
                   Img = single(imread(imgfile));
                   if size(Img, 3) == 1
                      Img = cat(3, Img, Img,Img);
                   end;
                   [F,D] = vl_sift(single(rgb2gray(Img)));
                   featsift = [featsift,D];
                   framecount(count)=size(D,2);
                   count = count+1;
               end;
               save([siftpath,Acts(i).name,'_',instances(j).name,'.mat'],'featsift','framecount');
            end;
        end;
    end;
end;
