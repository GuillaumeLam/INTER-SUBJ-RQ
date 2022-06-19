% script for data.mat => py_compatible_data.mat
% todo: get relevant info from data.mat

data_mat = load('./data.mat').('data');

% iter thru all patients
ids = fieldnames(data_mat);
for p=1:numel(ids)
    sensors = data_mat.(ids{p});    % redefine struct for simplicity
    for t=1:57                      % itet thru 57 trials to add each
        if t < 4                    % skip calib trials
            continue
        else
            y = sensors.('trunk').('Surface')(t);

            ch = {'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z'};
            seg = {'trunk', 'thighL', 'thighR', 'shankL', 'shankR'};

            for s=1:length(seg)
                x(:,:,s)=cell2mat(table2array(sensors.(seg{s})(t,ch)));
            end

            x = permute(x,[2,3,1]);
            entry = {p, x , y};

            % split x based on gait events and add each 101x5x6 gait cycle
            % append entry as new cell array row
        end
    end
end


t = zeros(543,5,6);
entry1 = {1,t, 'cs'};
entry2 = {2,t,'fe'};
simplified_data = [entry1;entry2];
save('py_compatible_data.mat', 'simplified_data');
