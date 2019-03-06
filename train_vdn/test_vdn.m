param.index_path = fullfile(pwd, 'test_data_split.mat');
param.data_path = fullfile(fileparts(pwd), 'dataset\');
param.caffe_path = fullfile(fileparts(pwd), 'caffe');
param.log_path = fullfile(fileparts(pwd), 'caffe', 'log');
param.model_path = fullfile(fileparts(pwd), 'model\');
param.caffemodel_path = fullfile(pwd, 'train_mod\');
param.category_number = 40;
param.model_number = 800;
param.model_type = 'cdu';
param.model_iter = 78000;
param.view_count = 10;
param.feature_size = 1024;
param.b = 104;
param.g = 117;
param.r = 123;

load(param.index_path);
label_index = repmat(struct('start',0), param.category_number, 1);
last_category = -1;
m = 1;
for n = 1 : param.model_number
    if (last_category ~= test_data_split(n * param.view_count).label)
        label_index(m).start = n - 1;
        if (m > 1)
            label_index(m-1).end = n - 1;
        end
        last_category = test_data_split(n * param.view_count).label;
        m = m + 1;
    end
end
label_index(m-1).end = param.model_number;

curpath = pwd;
addpath(param.caffe_path);
cd(param.caffe_path);
caffe.reset_all;
caffe.init_log(param.log_path);
cd(curpath);
caffe.set_mode_gpu;
deploy = [param.model_path, 'test_', param.model_type, '.prototxt'];
caffemodel = [param.caffemodel_path, 'vdn_', param.model_type, '_iter_', int2str(param.model_iter), '.caffemodel'];
net = caffe.get_net(deploy,'test');
net.copy_from(caffemodel);

data_shape = net.blobs('data').shape;
batch_size = data_shape(4);
batch_data = zeros(data_shape,'single');
matrix = zeros([param.feature_size,param.model_number],'single');
model_count = batch_size / param.view_count;

data_pointer = 1;
while true
    data_end = min(data_pointer+model_count-1,param.model_number);
    data_count = data_end - data_pointer + 1;
    if data_count == 0
        break;
    end
    if data_count ~= model_count
        data_pointer = data_end - model_count + 1;
    end
    for ii = 1 : data_count
        sel = data_pointer + ii - 1;
        for jj = 1 : param.view_count
            pos = (jj-1) * model_count + ii;
            ind = sel * param.view_count - param.view_count + jj;
            image = imread([param.data_path,test_data_split(ind).path]);
            for w = 1 : 224
                if sum(image(w,:)) == 0
                    continue;
                end
                u = w;
                break;
            end
            for w = 1 : 224
                if sum(image(225-w,:)) == 0
                    continue;
                end
                d = 225-w;
                break;
            end
            for h = 1 : 224
                if sum(image(:,h)) == 0
                    continue;
                end
                l = h;
                break;
            end
            for h = 1 : 224
                if sum(image(:,225-h)) == 0
                    continue;
                end
                r = 225-h;
                break;
            end
            
            if ~isempty(l)
                if r - l > d - u
                    width = single(224) / single(r - l) * single(d - u);
                    width = uint16(width);
                    if (width == 0)
                        width = 1;
                    end
                    temp = imresize(image(u:d,l:r),[width 224]);
                    tempdata = zeros([224,224],'uint8');
                    margin = uint16((224 - width) / 2);
                    tempdata(margin+1:margin+width,1:224) = temp;
                else
                    width = single(224) / single(d - u) * single(r - l);
                    width = uint16(width);
                    if (width == 0)
                        width = 1;
                    end
                    temp = imresize(image(u:d,l:r),[224 width]);
                    tempdata = zeros([224,224],'uint8');
                    margin = uint16((224 - width) / 2);
                    tempdata(1:224,margin+1:margin+width) = temp;
                end
            else
                tempdata = image;
            end
            
            tempdata = permute(tempdata, [2, 1, 3]);
            tempdata = single(tempdata);
            batch_data(:,:,1,pos) = tempdata-param.b;
            batch_data(:,:,2,pos) = tempdata-param.g;
            batch_data(:,:,3,pos) = tempdata-param.r;
        end
    end
    net.blobs('data').set_data(batch_data);
    net.forward_prefilled;
    temp = net.blobs('pool5/7x7_s1').get_data;
    matrix(:,data_pointer:data_end) = temp;
    data_pointer = data_end + 1;
end
matrix = matrix';

features = matrix;
for i = 1 : 800
    features(i, :) = features(i, :) / norm(features(i, :), 2);
end
DistMatrix = features * features';
fp = fopen('DistMatrix.txt','w');
fprintf(fp,'%.6f ',DistMatrix);
MAP_AUC;