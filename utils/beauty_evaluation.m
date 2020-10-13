% evaluation
dataset = 'scut-fbp5500';
cr = 0.5;
loss = {'l1', 'l2', 'sm', 'rankbce', 'rankmse', 'ldkl', 'ldklexpl1'};
for i = 1:numel(loss)
    switch loss{i}
        case 'ldklexpl1'
            path =  fullfile(dataset, 'max2_avg7', [loss{i}, '_thinvggbn_maxavg_msceleb1m_mt_CR',num2str(cr),'_Aug_Lambda_1/result.mat']);
        otherwise
            path =  fullfile(dataset, 'max2_avg7', [loss{i}, '_thinvggbn_maxavg_msceleb1m_CR',num2str(cr),'_Aug/result.mat']);
    end
    load(path)
    pred = (pred1+pred2)./2;
%     pred = pred1;
    validlabel = class;
    validsigma = sigma;
    
    mae = mean(abs(pred-validlabel));
    rmse = sqrt(mean((pred-validlabel).^2));
    sigma = validsigma;
    inds = find(sigma ==0);
    if ~isempty(inds)
        sigma(inds) = 1e-10;
    end
    error = mean(1-exp(-0.5*((pred-validlabel)./sigma).^2));
    ro = corrcoef(pred,validlabel);
    fprintf('loss:%s, mae:%.3f, rmse:%.3f, pc:%.3f, error:%.3f\n', loss{i}, mae, rmse, ro(1,2), error);
end

% % split1 cr = 0.5
% loss:ldkl, mae:0.178, rmse:0.237, pc:0.941, error:0.059
% loss:ldklexpl1, mae:0.173, rmse:0.229, pc:0.943, error:0.055
% % cr = 0.25
% loss:ldkl, mae:0.193, rmse:0.254, pc:0.933, error:0.067
% loss:ldklexpl1, mae:0.185, rmse:0.243, pc:0.935, error:0.062
% % split2 cr = 0.5
% loss:ldkl, mae:0.182, rmse:0.243, pc:0.940, error:0.061
% loss:ldklexpl1, mae:0.172, rmse:0.233, pc:0.940, error:0.057
% % cr = 0.25
% loss:ldkl, mae:0.190, rmse:0.253, pc:0.934, error:0.067
% loss:ldklexpl1, mae:0.176, rmse:0.239, pc:0.938, error:0.059
% % split3 cr = 0.5
% loss:ldkl, mae:0.188, rmse:0.250, pc:0.936, error:0.066
% loss:ldklexpl1, mae:0.179, rmse:0.240, pc:0.939, error:0.060
% % cr = 0.25
% loss:ldkl, mae:0.201, rmse:0.261, pc:0.930, error:0.072
% loss:ldklexpl1, mae:0.189, rmse:0.249, pc:0.934, error:0.065
% % split4 cr = 0.5
% loss:ldkl, mae:0.187, rmse:0.245, pc:0.938, error:0.064
% loss:ldklexpl1, mae:0.178, rmse:0.235, pc:0.940, error:0.059
% % cr = 0.25
% loss:ldkl, mae:0.198, rmse:0.259, pc:0.928, error:0.070
% loss:ldklexpl1, mae:0.189, rmse:0.249, pc:0.934, error:0.066
% % split5 cr = 0.5
% loss:ldkl, mae:0.175, rmse:0.226, pc:0.948, error:0.055
% loss:ldklexpl1, mae:0.169, rmse:0.219, pc:0.948, error:0.051
% % cr = 0.25
% loss:ldkl, mae:0.184, rmse:0.239, pc:0.940, error:0.061
% loss:ldklexpl1, mae:0.178, rmse:0.231, pc:0.942, error:0.058




[v, id] = sort(validlabel)

figure
plot(v, 'r.')
hold on
plot(pred(id), 'g.')