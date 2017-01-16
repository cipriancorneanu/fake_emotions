% Main reading script for BU4DFE

path2data = '/Users/cipriancorneanu/Research/data/fake_emotions/FakeEmotion_Videos/';
path2save = '/Users/cipriancorneanu/Research/data/fake_emotions_proc/';

% Add dependecies
addpath(genpath('../demo'));
addpath(genpath('../sdm'));

% Define reader class
fEmoReader = FakeEmotionsReader(path2data, path2save);

% Read and save
fEmoReader.Read();


