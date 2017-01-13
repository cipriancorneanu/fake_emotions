function plotSampleImages(path, dataset)
    for i = 10:50
        img = imread([path dataset(i).fpath]);
        if min(dataset(i).rect) <= 0 || dataset(i).rect(4)+dataset(i).rect(2) > size(img,2) ||dataset(i).rect(3)+dataset(i).rect(1) > size(img,1)
            continue
        end
        
        img = img(dataset(i).rect(2):dataset(i).rect(4)+dataset(i).rect(2), dataset(i).rect(1):dataset(i).rect(3)+dataset(i).rect(1), :);
        scfac = 300/mean(dataset(i).rect(3:4));
        landmarks = bsxfun(@minus, dataset(i).landmarks, dataset(i).rect(1:2)) * scfac;
        
        figure;
        imshow(imresize(img, scfac));
        hold on;
        scatter(landmarks(:,1), landmarks(:,2), 'g');
        hold off;
    end
end