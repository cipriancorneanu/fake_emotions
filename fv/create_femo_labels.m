clc;
clear all;
close all;

actors = {'Laur', 'Alena', 'Iiris', 'Ahmed', 'Andreas', 'Anton', 'Darwin', 'Dana', ...
    'Elmar', 'frances', 'Hassan', 'Iiris', 'Ivan', 'kaisa', 'KarlGregori', 'Kirill', 'Laura', ...
    'LauraJogede', 'Mari-liis', 'Lucas', 'AlexanderMakarov', 'Aleksander', 'Mate', ...
    'Merilin', 'Nikita', 'Nina', 'Pavel', 'Pejman', ...
    'Remo', 'Richard', 'Suman', 'Roxanne', 'Reka', 'Zemaio', 'Vladimir', 'Vladimiz', 'Chris',...
    'nana', 'sinle', 'yiiri', ...
    'age', 'Anne', 'Teddy', 'Asif', 'Rezwan', 'Sameer', 'Reena', 'Toomas', 'Lembit', 'Yeh',...
    'Umesh', 'Helen', 'Karl', 'Aiirin'};

key_emo = {'N2Sur', 'N2S', 'N2H', 'N2D', 'N2C', 'N2A', 'S2N2H', ....
            'H2N2D', 'H2N2C', 'H2N2A', 'D2N2Sur', 'H2N2S', };

v_emo = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
emotions = containers.Map(key_emo, v_emo);

for i = 1:2
    act = actors(i);
    for j = 1:10
        emo = key_emo(j);
        key = strcat(emo, act)
    end

end




