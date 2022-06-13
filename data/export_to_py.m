% script for data.mat => py_compatible_data.mat
% todo: get relevant info from data.mat

t = zeros(543,5,6);
entry1 = {1,t, 'cs'};
entry2 = {2,t,'fe'};
data = [entry1;entry2];
save('py_compatible_data.mat', 'data');
