% Main reading script for BU4DFE

path2data = '/Users/cipriancorneanu/Research/data/fake_emotions_proc/'
path2save = '/Users/cipriancorneanu/Research/data/fake_emotions_proc_res/';

% Add dependecies
addpath(genpath('../demo'));
addpath(genpath('../sdm'));

% Define reader class
fEmoReader = FakeEmotionsReader(path2data, path2save);

% Read and save
fEmoReader.Read();


