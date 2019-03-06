param.index_path = fullfile(pwd, 'train_data_split.mat');
param.data_path = fullfile(fileparts(pwd), 'dataset\');
param.caffe_path = fullfile(fileparts(pwd), 'caffe');
param.log_path = fullfile(fileparts(pwd), 'caffe', 'log');
param.model_path = fullfile(fileparts(pwd), 'model\');
param.pre_train_model_path = fullfile(fileparts(pwd), 'googlenet_bn.caffemodel');
param.caffemodel_path = fullfile(pwd, 'train_mod\');
param.category_number = 40;
param.model_number = 3183;
param.model_type = 'pdu';
param.view_count = 10;
param.gpu_num = 1;
param.gpu_id = 3;
param.snapshot = 1000;
param.max_iter = 80000;
param.plot_interval = 20;
param.b = 104;
param.g = 117;
param.r = 123;

load (param.index_path);
label_index = repmat(struct('start',0), param.category_number, 1);
last_category = -1;
m = 1;
for n = 1 : param.model_number
    if (last_category ~= train_data_split(n * param.view_count).label)
        label_index(m).start = n - 1;
        if (m > 1)
            label_index(m-1).end = n - 1;
        end
        last_category = train_data_split(n * param.view_count).label;
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
solver = [param.model_path, 'solver_', param.model_type, '.prototxt'];
caffe_solver = caffe.get_solver(solver,param.gpu_id);
caffe_solver.nets{1}.copy_from(param.pre_train_model_path);
data_shape = caffe_solver.nets{1}.blobs('data').shape;
batch_size = data_shape(4);
batch_data = zeros(data_shape,'single');
batch_label1 = zeros(1,1,1,batch_size,'single');
batch_label2 = zeros(1,1,1,batch_size / param.view_count,'single');
model_count = batch_size / param.view_count;
inputs = cell(1,param.gpu_num);
for i = 1:param.gpu_num
    inputs{i}{1} = batch_data;
    inputs{i}{2} = batch_label1;
    inputs{i}{3} = batch_label2;
end

x = [];
y = [];

while true
    for i = 1 : param.gpu_num
        data_sel = randperm(param.category_number,model_count/2);
        for ii = 1 : model_count/2
            if(rand >= 0.5)
                judge = 1;
            else
                judge = 0;
            end
            
            model_start = label_index(data_sel(ii)).start;
            model_end = label_index(data_sel(ii)).end;
            sel_models = model_start + randperm(model_end - model_start,2);
            sel_model = sel_models(1);
            sel_start = param.view_count * (sel_model - 1);
            inputs{i}{3}(1,1,1,ii * 2 - 1) = data_sel(ii) - 1;
            
            %first model
            for jj = 1 : param.view_count
                pos = (jj-1) * model_count+ii*2-1;
                ind = sel_start + jj;
                inputs{i}{2}(1,1,1,pos) = data_sel(ii) - 1;
                image = imread([param.data_path,train_data_split(ind).path]);
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
                inputs{i}{1}(:,:,1,pos) = tempdata-param.b;
                inputs{i}{1}(:,:,2,pos) = tempdata-param.g;
                inputs{i}{1}(:,:,3,pos) = tempdata-param.r;
            end
            
            %adjust pair label
            if judge == 1
                sel_model = sel_models(2);
                sel_start = param.view_count * (sel_model - 1);
                inputs{i}{3}(1,1,1,ii * 2) = data_sel(ii) - 1;
            else
                sel_model = randperm(param.model_number,1);
                sel_start = param.view_count * (sel_model - 1);
                inputs{i}{3}(1,1,1,ii * 2) = train_data_split(sel_model*param.view_count).label - 1;
            end
            
            %second model
            for jj = 1 : param.view_count
                pos = (jj-1) * model_count+ii*2;
                ind = sel_start + jj;
                inputs{i}{2}(1,1,1,pos) = inputs{i}{3}(1,1,1,ii * 2);
                image = imread([param.data_path,train_data_split(ind).path]);
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
                inputs{i}{1}(:,:,1,pos) = tempdata-param.b;
                inputs{i}{1}(:,:,2,pos) = tempdata-param.g;
                inputs{i}{1}(:,:,3,pos) = tempdata-param.r;
            end
        end
    end
    caffe_solver.set_input_data(inputs);
    caffe_solver.step(1);
    
    iter = caffe_solver.iter;
    loss1 = caffe_solver.nets{1}.blobs('loss1/loss').get_data;
    loss2 = caffe_solver.nets{1}.blobs('loss2/loss').get_data;
    loss3 = caffe_solver.nets{1}.blobs('loss3/loss').get_data;
    acc = caffe_solver.nets{1}.blobs('loss3/top-1').get_data;
    fprintf('iter:%-10d,loss1:%f\tloss2:%f\tloss3:%f\tacc:%f\n',iter,loss1,loss2,loss3,acc);
    
    if mod(iter,param.plot_interval) == 0
        x = [x iter];
        y = [y loss3];
        plot(x,y);
        drawnow;
    end
    if mod(iter,param.snapshot) == 0
        caffe_solver.nets{1}.save([param.caffemodel_path, 'vdn_', param.model_type, '_iter_', int2str(iter), '.caffemodel']);
    end
end