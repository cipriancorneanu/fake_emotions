function [] = extractNoMissingsData()
    data = [];
    for i = 1:10
        tdata = load(['data/aflw_fold_' num2str(i)]);
        tdata = tdata.data;

        indexs = ~squeeze(sum(sum(cat(3, tdata.landmarks2d), 2) == 0, 1));
        data = [data ; tdata(indexs)];
    end
    
    save('data/aflw_nomissings.mat', 'data');
end