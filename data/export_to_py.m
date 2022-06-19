% script for data.mat => py_compatible_data.mat
% todo: get relevant info from data.mat

simplified_data = smpl_dataset();

save('py_compatible_data.mat', 'simplified_data');
