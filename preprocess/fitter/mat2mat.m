
path = '/Users/cipriancorneanu/Research/data/fake_emotions/geoms/';

% Define reader class
fEmoReader = FakeEmotionsReader(path, path);

% Read and save
files = fEmoReader.GetFiles(path, '*.mat');

for i=1:length(files)
   load(strcat(path, files{i})); 
   save(strcat(path, files{i}),'out');
end