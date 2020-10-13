CUDA_VISIBLE_DEVICES=8 th
require 'nn'
require 'cunn'
require 'cudnn'
require 'torchx'
ffi = require 'ffi'
require 'image'
t = require 'datasets/transforms'
matio = require 'matio'
opt = {}
opt.nGPU =1
opt.gen = "/mnt/data3/gaobb/projects/DLDL-v2/gen"
opt.dataset = 'scut-fbp'
opt.dataset = 'morph'
--opt.dataset = 'chalearn16'
--opt.dataset = 'cfd'
opt.nThreads = 10
opt.loss = 'ldklexpl1'
-- Model loading
rootPath = paths.concat("/mnt/data3/gaobb/projects/DLDL-v2/",opt.dataset,'max2_avg7',opt.loss.."_thinvggbn_maxavg_msceleb1m_mt_CR0.5_Aug_Lamda_1")
local labelset
if opt.dataset == 'scut-fbp' then
   labelset = torch.range(1, 5, 0.1):float()
elseif opt.dataset == 'cfd' then
   labelset = torch.range(1, 7, 0.1):float()
else
   labelset = torch.range(0, 100, 1):float()
end

if opt.loss == 'l1' or opt.loss == 'l2' then
   outDim  = 1
elseif opt.loss == 'rankbce' or opt.loss == 'rankmse' then
   outDim  = labelset:size(1) -1
elseif opt.loss == 'sm' or opt.loss == 'ldkl' or 'ldklexpl1' then
   outDim = labelset:size(1)
end
opt.tensorType = 'torch.CudaTensor'
modelPath = paths.concat(rootPath, 'model_60.t7')
assert(paths.filep(modelPath), 'File not found: ' .. modelPath)
print('Loading model from file: ' .. modelPath)
net = torch.load(modelPath):type(opt.tensorType)
--print(net)
-- Data loading
--trainLoader, valLoader = DataLoader.create(opt)
dataPath = paths.concat(opt.gen, opt.dataset .. '.t7')
data = torch.load(dataPath)
print(data)
-- forward
timer = torch.Timer()
dataTimer = torch.Timer()
dataTime = dataTimer:time().real
nCrops =1
meanstd = {
   mean = { 0.5958, 0.4637, 0.4065 },
   std = {  0.2693, 0.2409, 0.2352 },
}  
pred = torch.FloatTensor(data.val.imagePath:size(1)):fill(0)           
trans = {}
trans =  t.Compose{
         t.Scale(224),
         t.CenterCrop(224),
         t.ColorNormalize(meanstd),}
net:evaluate()
te = trans
opt.occ = 2
if opt.occ == 1 then
   dh = torch.range(1,224,224)
   dw = torch.range(1,224,32)
elseif opt.occ == 2 then
  dh = torch.range(1,224,32)
  dw = torch.range(1,224,224)
elseif opt.occ ==3 then 
  dh = torch.range(1,224,32)
  dw = torch.range(1,224,32) 
end
class = data.val.imageClass:clone()
sigma = data.val.imageSigma:clone()
maes = torch.FloatTensor(dh:numel(),dw:numel())
for h = 1, dh:numel() do
    for w =1, dw:numel()  do
        for i = 1, data.val.imagePath:size(1) do
            imgpath = paths.concat(data.basedir, ffi.string(data.val.imagePath[i]:data())) 
            img = image.load(imgpath, 'float')
            fimg = image.hflip(img)
            img1 = te(img)
            fimg1 = te(fimg)
            if opt.occ ==1 then 
               img1[{{},{dh[h],dh[h]+223},{dw[w],dw[w]+31}}]=0
               fimg1[{{},{dh[h],dh[h]+223},{dw[w],dw[w]+31}}]=0
            elseif opt.occ == 2 then
               img1[{{},{dh[h],dh[h]+31},{dw[w],dw[w]+223}}]=0
               fimg1[{{},{dh[h],dh[h]+31},{dw[w],dw[w]+223}}]=0
            elseif opt.occ == 3 then
               img1[{{},{dh[h],dh[h]+31},{dw[w],dw[w]+31}}]=0
               fimg1[{{},{dh[h],dh[h]+31},{dw[w],dw[w]+31}}]=0
            end
            imt = torch.FloatTensor(2,3,224,224)
            imt[{{1},{},{}}] = img1:clone()
            imt[{{2},{},{}}] = fimg1:clone()
            outs = net:forward(imt:cuda())
            local out1, out2 = unpack(outs)
            pred[i] =  out2:mean()
        end
        -- evaluation
        maes[{{h},{w}}] = (pred-class):abs():mean()
     end
end
matio.save(opt.dataset..opt.occ..'occ.mat', maes)

--[[matio.save(opt.dataset..'occh.mat', maes)

matio.save(opt.dataset..'occhw.mat', maes)


inds = torch.LongTensor(torch.find(sigma, 0))
if inds:dim() >=1 then
   sigma:indexFill(1, inds:long(), 1e-10)
end

local minl, maxl = labelset:min(), labelset:max()
for iter =  1, 2 do
   if opt.loss == 'l1' or opt.loss == 'l2' then
      -- regression
      f = (maxl-minl)/(1+1)
      result.pred[iter] =  ((result.score[iter] +1):mul(f) + minl)
   elseif opt.loss == 'sm' or opt.loss == 'ldkl' then
      -- dex, dldl, 
      result.pred[iter]  = result.score[iter]:float()*(labelset)
   elseif opt.loss == 'rankbce' or opt.loss == 'rankmse' then
      -- rank
      local sumcount = (result.score[iter]:float():ge(0.5):sum(2)):float() + 1
      result.pred[iter]  = labelset:index(1, sumcount:long():squeeze())
      --result.pred[iter]  = result.score[iter]:float():ge(.5):sum(2):float()
   elseif opt.loss == 'ldklexpl1' then
      result.pred[iter]  = result.scoreb[iter]:float()
   end
end


for iter = 1, 3 do
    if iter == 3 then
       pred = (result.pred[1]:squeeze() + result.pred[2]:squeeze())/2
       time = ((result.runTime[1] + result.runTime[2])/2):mean()
    else 
       pred = result.pred[iter]:squeeze()
       time = result.runTime[iter]:mean()
    end
    mae = (pred-class):abs():mean()
    cs3 = (pred-class):abs():le(3):sum()/pred:size(1)*100
    cs5 = (pred-class):abs():le(5):sum()/pred:size(1)*100
    cs8 = (pred-class):abs():le(8):sum()/pred:size(1)*100
    error = (1 - (pred - class):cdiv(sigma):pow(2):mul(-0.5):exp()):mean()
    if iter ==1 then
       print(('-- %s loss, labelStep %.2f'):format(opt.loss, opt.labelStep))
    end
    if iter == 3 then
       print(('fusion    | mae %.3f, error %.3f, cs3 %.3f, cs5 %.3f, cs8 %.3f, runtime %.5fs'):format(mae, error, cs3, cs5, cs8, time));
    else 
       print(('%d-th iter | mae %.3f, error %.3f, cs3 %.3f, cs5 %.3f, cs8 %.3f, runtime %.5fs'):format(iter, mae, error, cs3,cs5,cs8, time));
    end
end
result.class = class
result.sigma = sigma
torch.save(paths.concat(rootPath,'result.t7'), result)
matio.save(paths.concat(rootPath,'result.mat'), {class= result.class, 
                          sigma = result.sigma,
                          pred1 = result.pred[1],
                          pred2 = result.pred[2],
                          runTime1 = result.runTime[1], 
                          runTime2 = result.runTime[2],
                          score1 = result.score[1],
                          score2 = result.score[2],
                          })
--matio.save(paths.concat(rootPath,'result.mat'), result)
--[[      
opt = {}
datasets = {'chalearn15', 'chalearn16', 'morph'}
loss = {'l1', 'l2', 'sm', 'rankbce', 'rankmse', 'ldkl'}
dataset = datasets[1]
loss = {'ldklexpl1'}
for l= 1, #loss do
opt.loss = loss[l]
--result = torch.load(dataset..'/max2_avg7/'..opt.loss..'_thinvggbn_maxavg_msceleb1m_CR0.5_Aug_Lambda_0.01/result.t7')
result = torch.load(dataset..'/max2_avg7/'..opt.loss..'_thinvggbn_maxavg_msceleb1m_mt_CR0.5_Aug_Lambda_0.1/result.t7')

opt = {}
loss = {'ldkl'}
opt.loss = loss[1]
require 'torchx'
l =1 
opt.loss = 'ldkl'
result = torch.load('result.t7')


class = result.class
sigma = result.sigma
inds = torch.LongTensor(torch.find(sigma, 0))
if inds:dim() >=1 then
   sigma:indexFill(1, inds:long(), 1e-10)
end
for iter =  1, 2 do
   if opt.loss == 'l1' or opt.loss == 'l2' then
      -- regression
      f = (100-0)/(1+1)
      result.pred[iter] =  ((result.score[iter] +1):mul(f) + 0)
   elseif opt.loss == 'sm' or opt.loss == 'ldkl' then
      -- dex, dldl, 
      result.pred[iter]  = result.score[iter]:float()*(torch.range(0,100):float())
   elseif opt.loss == 'rankbce' or opt.loss == 'rankmse' then
      -- rank
      result.pred[iter]  = result.score[iter]:float():ge(.5):sum(2):float()
   elseif opt.loss == 'ldklexpl1' then
      result.pred[iter]  = result.scoreb[iter]:float()
   end
end
for iter = 1, 3 do
    if iter == 3 then
       pred = (result.pred[1]:squeeze() + result.pred[2]:squeeze())/2
       time = ((result.runTime[1] + result.runTime[2])/2):mean()
    else 
       pred = result.pred[iter]:squeeze()
       time = result.runTime[iter]:mean()
    end
    mae = (pred-class):abs():mean()
    cs3 = (pred-class):abs():le(3):sum()/pred:size(1)*100
    cs5 = (pred-class):abs():le(5):sum()/pred:size(1)*100
    cs8 = (pred-class):abs():le(8):sum()/pred:size(1)*100
    error = (1 - (pred - class):cdiv(sigma):pow(2):mul(-0.5):exp()):mean()
    if iter ==1 then
       print(('-- %s loss'):format(opt.loss))
    end
    if iter == 3 then
       print(('fusion    | mae %.3f, error %.3f, cs3 %.3f, cs5 %.3f, cs8 %.3f, runtime %.5fs'):format(mae, error, cs3, cs5, cs8, time));
    else 
       print(('%d-th iter | mae %.3f, error %.3f, cs3 %.3f, cs5 %.3f, cs8 %.3f, runtime %.5fs'):format(iter, mae, error, cs3,cs5,cs8, time));
    end
end
end]]

