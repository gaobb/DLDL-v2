

pred1 = score1*(0:100)'; 

pred2 = score2;
pred1 - pred2

[~, pred2] = max(score, [],2)
pred2 = score*(0:100)'; 

pred2 = sum(score>0.5, 2)+ 1;

pred2 = score2*(1:100)'; 

pred1 = sum(score>0.5, 2)+ 1;
score(1,:)


%% gen rank gt on training set
% method1
for c = 1:100
    ind = find(class==c);
    if isempty(ind)
       pdf = normpdf(1:100, c, 2);
       pdf = pdf./sum(pdf);
       temp = 1 - cumsum(pdf);
       rank_gt(c,:) = temp(1:end-1);
    elseif length(ind) == 1
       rank_gt(c,:) = agecode(ind, :);
    elseif length(ind) > 1
       rank_gt(c,:) = mean(agecode(ind, :));
    end
end
% method2
rank_gt = triu(-ones(100,99)) +1;


%% gen ld gt on training set
% method1
for c = 1:100
    ind = find(class==c);
    if isempty(ind)
       pdf = normpdf(1:100, c, 4);
       pdf = pdf./sum(pdf);
       rank_gt(c,:) = pdf;
    elseif length(ind) == 1
       rank_gt(c,:) = agecode(ind, :);
    elseif length(ind) > 1
       rank_gt(c,:) = mean(agecode(ind, :));
    end
end

% method2
for c = 1:100
    pdf = normpdf(1:100, c, 2);
    pdf = pdf./sum(pdf);
    rank_gt(c,:) = pdf;
end


%% kl
for n = 1:size(score, 1)
    [~,pred(n,1)] = min(sum(bsxfun(@times, -log(score(n,:)), rank_gt), 2));
end

%% mse rank
    score(score>0.98)=1;
    score(score<0.02)=0;
for n = 1:size(score, 1)
    [~,pred(n,1)] = min(sum(bsxfun(@minus, score(n,:), rank_gt).^2, 2));
end

pred2 = pdf*(0:100)'

[~, pred2] = max(score1, [],2)
pred1 = pred2-1

predld = score2(:,1:101);
predrank = score1(:,102:end);

predl = bsxfun(@times, predld, 1./sum(predld, 2)) *(0:100)';
[~, predl] = max(predld, [], 2);
predl = predl -1;

pred = (pred1 + pred2)./2;


dif = pred2-class;
mae = mean(abs(dif));
sigma(find(sigma==0)) = 1e-10;
error = mean(1- exp(-0.5.*(dif./sigma).^2));
fprintf('mae: %.4f, e-error: %.4f\n', mae, error);

plot(predl -predr)
pred = (pred1+pred2)./2

scores = bsxfun(@times, score, 1./sum(score,2));
pred2 = scores*(0:100)'; 
[~, pred2] = max(score, [], 2);
pred2 = pred2 -1;

pred2 = score*(1:100)';

pred1 =  sum(score1>0.5, 2);
pred1 =  sum(score >0.5, 2);

pred2 = score1*(0:100)'; 

[pred2 , score2, class]
figure
[v, ind] = sort(class);
plot(pred2(ind), 'g.');
hold on
plot(v, 'r');
grid on


plot(pred2(ind) - v, 'g.');


for i = 1:1136
M(i,:) = abs(diff(1-double([1, score(i,:), 0]),1));
end

[~, pred2] = max(M, [],2)


M = bsxfun(@times, M , 1./sum(M,2));
pred2 = M*(1:100)'; 
[~, pred2] = max(M, [],2)


ind(1974)


[v,pred2(ind) ]

plot(diff(ind))
x = 0:0.001:100;

plot(-x);
hold on
plot((-x + 1 -log(x+1)
plot(-(x+sqrt(x)))
legend('kl', 'rekl')



x1 = 100:-1:-100;
y1 = 1./(1+exp(-x1))


x2 = [-100:1:0, 0:-1:-100]
y2 = exp(x1);


figure
plot(x1)
hold on
plot(x2)


figure 
plot(y1)
hold on
plot(y2)


imdb = load('/mnt/data3/gaobb/image_data/image_faces/age_faces/MTCNN_Google/MTCNN_clean_imdb.mat');
ind = 1:1000
for i = 100:200
    img_path = fullfile('/mnt/data3/gaobb/image_data/image_faces/age_faces/MTCNN_Google/Align.5GoogleClean', imdb.images.name{ind(i)});
    img = imread(img_path);
    
    imshow(img);
    title(sprintf('%.f, %.f', class(ind(i)), pred2(ind(i))));
    pause
end




x = normpdf(1:100, 1, 2);
x = x./sum(x);
for i = 1:99
    ps(i) = 1 - sum(x(1:i));
end


pdf = normpdf(1:100, 50, 5);
cdf = cumsum(pdf)

sum(pdf)
plot(1-cdf)



pdf = normpdf(1:100, 50, 5);
pdf = pdf./max(pdf);
plot(pdf)



pdf = bsxfun(@times, agecode, 1./sum(agecode,2));
pdf = bsxfun(@times, predld, 1./sum(predld,2));

ex1 = pdf*((0:100)'.^2);
ex2 = (pdf*(0:100)').^2;
histogram(ex1-ex2, 100)

x = 38
pdf = normpdf(0:100, x, 3)
y = x/50 -1
pdf* (-1:2/100:1)'




perma = randperm(5613)
permb = randperm(5613)


labela = class(perma)
labelb = class(permb)


label = find((labela > labelb)==1)

labela > labelb










A =[4.891  4.862 ;
4.898    4.851;
4.781    4.781;
5.067  5.006;
4.770   4.758;]

A =[5.148   4.891 
 4.883   4.700 
 4.938    4.770
 5.273    4.983
 5.469    5.197]
 A=[ 5.936    5.850
 5.569    5.473
  5.794   5.763
  5.691  5.607
  5.362   5.375]

A=[5.746   5.600;
5.934    5.769;
5.682    5.560;
5.811   5.741;
5.852    5.678;]

A= [ 4.592  4.651
  4.706   4.746
  4.757   4.827
  4.724   4.808 
  4.843   4.925]


A=[4.736  4.736
 4.947    4.947
  4.787    4.787 
  4.606   4.606
   4.842   4.842;]

A =[4.717    4.887
  4.817    4.875
   4.904    4.868
   4.871    4.869
   4.927    4.974]


A=[   4.820   5.027  
   4.970    5.171
   4.887    5.100
   4.886    5.083
   4.900    5.078];

A=[ 4.754  4.754
  4.665    4.665
  4.688    4.688 
  4.633    4.633
  4.833    4.833];

