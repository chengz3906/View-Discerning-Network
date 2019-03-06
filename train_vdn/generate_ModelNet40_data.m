%generate data config
param.file_path = fullfile(fileparts(pwd),'dataset','ModelNet40');
param.save_traindata_filename = 'train_data_split';
param.save_testdata_filename = 'test_data_split';
%generate split train/test data
subdir = dir(param.file_path);
%generate train data
fprintf('process train data start\n');
train_data_split = [];
train_data_split(1).path = '';
train_data_split(1).label = 0;
train_data_split(1).className = '';
index = 1;
for i = 3:length(subdir)
    sub_subdir = dir(fullfile(param.file_path, subdir(i).name, 'train'));
    random_list = randperm(length(sub_subdir)-2,min(length(sub_subdir)-2,80)) + 2;
    for j = 1:length(random_list)
        sub_sub_subdir = dir(fullfile(param.file_path,subdir(i).name, 'train', sub_subdir(random_list(j)).name));
        random_list1 = randperm(min(length(sub_sub_subdir)-2,10),10) + 2;
        for k = 1:length(random_list1)
            train_data_split(index).label = i-2;
            train_data_split(index).className = subdir(i).name;
            train_data_split(index).path = fullfile(param.file_path,subdir(i).name, 'train', sub_subdir(random_list(j)).name,sub_sub_subdir(random_list1(k)).name);
            index = index + 1;
        end
    end
    
end
save(strcat(param.save_traindata_filename,'.mat'),'train_data_split');
fprintf('process train data end\n');
%generate test data
fprintf('process test data start\n');
test_data_split = [];
test_data_split(1).path = '';
test_data_split(1).label = 0;
test_data_split(1).className = '';
index = 1;

for i = 3:length(subdir)
    sub_subdir = dir(fullfile(param.file_path, subdir(i).name, 'test'));
    random_list = randperm(length(sub_subdir)-2,min(length(sub_subdir)-2,20)) + 2;
    for j = 1:length(random_list)
        sub_sub_subdir = dir(fullfile(param.file_path,subdir(i).name, 'test', sub_subdir(random_list(j)).name));
        random_list1 = randperm(min(length(sub_sub_subdir)-2,10),10) + 2;
        for k = 1:length(random_list1)
            test_data_split(index).label = i-2;
            test_data_split(index).className = subdir(i).name;
            test_data_split(index).path = fullfile(param.file_path,subdir(i).name, 'test', sub_subdir(random_list(j)).name,sub_sub_subdir(random_list1(k)).name);
            index = index + 1;
        end
    end
    
end
save(strcat(param.save_testdata_filename,'.mat'),'test_data_split');
fprintf('process test data end\n');
