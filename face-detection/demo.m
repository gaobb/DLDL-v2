% face detection and aligment
addpath('/mnt/data3/gaobb/projects/MTCNN_face_detection_alignment/code/codes/MTCNNv2')

clear;
%list of images
% imglist=importdata('imglist.txt');
%minimum size of face
minsize=20;

%path of toolbox
caffe_path='/home/gaobb/Software/caffe-ea455eb/matlab';
current_path = pwd;
pdollar_toolbox_path='/mnt/data3/gaobb/projects/MTCNN_face_detection_alignment/toolbox';
caffe_model_path='/mnt/data3/gaobb/projects/MTCNN_face_detection_alignment/code/codes/MTCNNv2/model';
addpath(genpath(caffe_path));
addpath(genpath(pdollar_toolbox_path));

%use cpu
%caffe.set_mode_cpu();
gpu_id=15;
caffe.set_mode_gpu();	
caffe.set_device(gpu_id);

%three steps's threshold
threshold=[0.6 0.7 0.7];

%scale factor
factor=0.709;

%load caffe models
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
model_dir = strcat(caffe_model_path,'/det1.caffemodel');
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
model_dir = strcat(caffe_model_path,'/det2.caffemodel');
RNet=caffe.Net(prototxt_dir,model_dir,'test');	
prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
model_dir = strcat(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir =  strcat(caffe_model_path,'/det4.prototxt');
model_dir =  strcat(caffe_model_path,'/det4.caffemodel');
LNet=caffe.Net(prototxt_dir,model_dir,'test');
faces=cell(0);	

%% face detection
img_path = 'trump.jpg';

try
    img = imread(img_path);
catch
    system(['convert ', img_path, ' -colorspace RGB  ', '/tmp/temp.JPEG']); % convert CMYK into RGB
    img = imread('/tmp/temp.JPEG');
end

if size(img,3) ==1
    img = cat(3, img,img,img);
end

imgsize = size(img);
if min(imgsize(1:2)) > 1500
    scale = 1500./min(imgsize(1:2));
    img = imresize(img, scale);
else
    scale = 1;
end
    
[bboxes, points]= detect_face(img,minsize,PNet,RNet,ONet,LNet,threshold,false,factor);
if isempty(bboxes)
    [h,w,~] = size(img);
    bboxes = [1,1, w, h, -inf];
    points = [];
end
if scale ~= 1
    bboxes = bboxes./scale;
    points = points./scale;
end

numbox=size(bboxes,1);
%% Alignment
load('/mnt/data3/gaobb/projects/MTCNN_face_detection_alignment/code/codes/MTCNNv2/norm_mean_shape_setting2.mat');
imgSize = [224 224];
coord5points = bsxfun(@plus, norm_mean_shape*0.5, (imgSize+1)./2); %0.8
imgsize = size(img);
face_root = './trump';
mkdir(face_root)

for n =1:numbox
    keypoints = points(:,n);
    facial5points = double(reshape(keypoints(:,1),5,2))';
    Tfm = cp2tform(facial5points',coord5points,'nonreflective similarity');
    align_img1 = imtransform(img,Tfm,'XData',[1 imgSize(2)],'YData',[1 imgSize(1)]);
    align_img = imtransform(img,Tfm,'XData',[1 imgSize(2)],'YData',[1 imgSize(1)],'XYScale',1);
    
    save_path = fullfile(face_root, [num2str(n),'.jpg']);
    imwrite(align_img, save_path);
end

figure


%% crop
for n =1:numbox
    crop_img = crop(img,bboxes(n,:));
    
    save_path = fullfile(face_root, [num2str(n),'crop.jpg']);
    imwrite(crop_img, save_path);
end


%show detection result
img = imread(img_path);
numbox=size(bboxes,1);
close all
figure
iptsetpref('ImshowBorder','tight');
imshow(img)
hold on
for n=1:numbox
    bbox  = bboxes(n,:);
    plot(points(1:5,n),points(6:10,n),'g.','MarkerSize',10);
    r=rectangle('Position',[bbox(1,1:2) bbox(1,3:4)-bbox(1,1:2)],'Edgecolor','m','LineWidth',3);
end
print(gcf,'-depsc',['trump','.eps'])


% geerate gif
imwrite(align_img, 'trump_gif.jpg');
hold on
for n=1:numbox
    bbox  = bboxes(n,:);
    plot(points(1:5,n),points(6:10,n),'g.','MarkerSize',10);
    r=rectangle('Position',[bbox(1,1:2) bbox(1,3:4)-bbox(1,1:2)],'Edgecolor','m','LineWidth',3);
    wh = (bbox(1,3:4)-bbox(1,1:2))/2;
    pos = bbox(1,1:4);
    text(double(pos(1)+ wh(1)-42), double(pos(2)-15),...
        sprintf('%.1f', pred_age(n)), ...
        'color', 'g', 'fontsize', 8);
    text(double(pos(1)+ wh(1)-42), double(pos(4)+15),...
        sprintf('%.2f', pred_att(n)), ...
        'color', 'g', 'fontsize', 8);
    if mod(n,10)==0 | n==numbox
        frame = getframe(1);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if n == 10
            imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
        else
            imwrite(imind,cm,filename,'gif','WriteMode','append');
        end
    end
end