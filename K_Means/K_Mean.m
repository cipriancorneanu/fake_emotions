close all
clear all
clc
%load ('koordinate.mat');% set of landmarks with all frames for one video file
III=[];
newset1=[];
fileList=dir('*.csv');
s=length(fileList);


for j=1:s
     temp=fileList(j).name;
     d = csvread(temp,20,10,[20,10,24,13]);
     
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
      II=[I1 I2 I3 I4];
      III=[III;II];
    
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
     
     numLines=4;
     m=0;
     for l=1:length(temp)
         if temp(l)=='_'
             m=l;
         end
     end
     fileIndex=str2double(temp(m+1:end-4));
     newIndex=numLines*(fileIndex)+1;
     newset1(newIndex:newIndex+numLines-1,:)=newset;
end
[m,n]=size(newset1);
C=cell(m,1);
C(:)=p(2,end);
result=vertcat(p(1,:),horzcat(num2cell(newset1),C));
xlswrite('Angry.xlsx',result)
xlswrite('Angry_indices_kmean.xlsx', III);