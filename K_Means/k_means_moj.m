close all
clear all
clc
%load ('koordinate.mat');% set of landmarks with all frames for one video file
newset1=[];
fileList=dir('*.csv');
s=length(fileList);
for i=1:s
     temp=fileList(i).name;
     [koordinate,p,q]= xlsread(temp);
     
     % Number of selected frames
     k=4;
         newset=zeros(k,136); %new set of selected frames which landmarks are closest to centroids
         koordinate=koordinate(:,1:136); %avoid label
        [IDX, centers, SUMD, D]  = kmeans(koordinate,k); %% Position of centroids

%N-by-P data matrix X into K clusters
  %%%  [IDX, C, SUMD, D] = kmeans(X, K) returns distances from each point
   % to every centroid in the N-by-K matrix D.
        [M1,I1]= min(D(:,1)); 
%find min distance for the first cluster and return the number of the frame
        newset(1,:)=koordinate(I1,:);
    
       [M2,I2]=min(D(:,2));
       %find min distance for the second cluster and return the number of the frame
       newset(2,:)=koordinate(I2,:);
    
       [M3,I3]=min(D(:,3));
       %find min distance for the third cluster and return the number of the frame
       newset(3,:)=koordinate(I3,:);

      [M4,I4]=min(D(:,4));
      newset(4,:)=koordinate(I4,:);
    
       %find min distance for the third cluster and return the number of the frame    


      %newset %%%4 selected frames for each video file


%Plotting faces with centroids-figure 1, Plot faces with landmarks that are
%closest to centroids
  for k=1:4
       x=zeros(k,68);
    y=zeros(k,68);
    for i=1:68
    x(k,i)=newset(k,i);
   end
   for i=69:136
         y(k,i)=newset(k,i);
   end
   
   %plot(x(k,:),-y(k,69:136));%Plotting faces with centroids
   %hold on;
  end
    
    %figure(2)
     for k=1:4
       x=zeros(k,68);
    y=zeros(k,68);
    for i=1:68
    x(k,i)=newset(k,i);
   end
   for i=69:136
         y(k,i)=newset(k,i);
   end
   
   
    
   %plot(x(k,:),-y(k,69:136)); %Plotting faces with landmarks that are
   %closest to centroids
   %hold on;
     end
     newset1=[newset1;newset];
end
% [mm nn]=size(p);
% newset2=zeros(mm,nn);
% newset2(1,:)=p(1,:);


