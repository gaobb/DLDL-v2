
require 'nn'
require 'cunn'
require 'cudnn'
require 'torchx'
ffi = require 'ffi'
require 'image'
t = require 'datasets/transforms'
matio = require 'matio'
function parse(arg)
   local cmd = torch.CmdLine()
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 DLDL+Exp Evaluation script')
   cmd:text('See https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md for examples')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
   cmd:option('-gen',        "/mnt/data3/gaobb/projects/DLDL-v2/gen",      'Path to save generated files')
   
   ------------- Data options ------------------------
   cmd:option('-nThreads',        10,     'number of data loading threads')
   cmd:option('-tensorType',     'torch.CudaTensor',   'Options: ld | histc  | kde')
   cmd:option('-dataset',        'chalearn15',     'number of data loading threads')
   cmd:option('-dataAug',        'true',     'Options: true or false')
   cmd:option('-lambda',          1,      'the hyper-parameter between kl and l1,Options: 1 or 0.1 or 0.01')
   cmd:option('-labelStep',       1,      'the interval of two neighborhood labels')
   ---------- Model options ----------------------------------
   cmd:option('-loss',         'ldkl',     'Options: kl or l1 or sm or sl1')
   cmd:option('-CR',           0.5,      'Compression rate: 1|1/2 | 1/4 | 1/8')
   cmd:text()
   local opt = cmd:parse(arg or {})
 return opt
end
opt = parse(arg)
--print(opt)

-- Model loading
if opt.loss == 'ldklexpl1' or opt.loss == 'ldklexpsmoothl1' then
   if opt.dataAug == 'true' then
   rootPath = paths.concat("/mnt/data3/gaobb/projects/DLDL-v2/",opt.dataset,'max2_avg7',opt.loss.."_thinvggbn_maxavg_msceleb1m_mt_CR"..opt.CR.."_Aug_Lambda_"..opt.lambda)
   else
   rootPath = paths.concat("/mnt/data3/gaobb/projects/DLDL-v2/",opt.dataset,'max2_avg7',opt.loss.."_thinvggbn_maxavg_msceleb1m_mt_CR"..opt.CR.."_Lambda_"..opt.lambda)
   end
elseif opt.loss == 'ldklexpl2' then
   if opt.dataAug == 'true' then
   rootPath = paths.concat("/mnt/data3/gaobb/projects/DLDL-v2/",opt.dataset,'max2_avg7',opt.loss.."_thinvggbn_maxavg_msceleb1m_mt_CR"..opt.CR.."_Aug_Lambda_"..opt.lambda)
   else
   rootPath = paths.concat("/mnt/data3/gaobb/projects/DLDL-v2/",opt.dataset,'max2_avg7',opt.loss.."_thinvggbn_maxavg_msceleb1m_mt_CR"..opt.CR.."_Lambda_"..opt.lambda)
   end
else
   if opt.dataAug == 'true' then
      if opt.labelStep ==1 or opt.labelStep ==0.1 then
      rootPath = paths.concat("/mnt/data3/gaobb/projects/DLDL-v2/",opt.dataset,'max2_avg7', opt.loss.."_thinvggbn_maxavg_msceleb1m_CR"..opt.CR.."_Aug")
      else
      rootPath = paths.concat("/mnt/data3/gaobb/projects/DLDL-v2/",opt.dataset,'max2_avg7', opt.loss.."_thinvggbn_maxavg_msceleb1m_CR"..opt.CR.."_Aug_sigma_Step"..opt.labelStep) 
      end
   else
      rootPath = paths.concat("/mnt/data3/gaobb/projects/DLDL-v2/",opt.dataset,'max2_avg7',opt.loss.."_thinvggbn_maxavg_msceleb1m_CR"..opt.CR)
   end
end

--labelset = torch.range(0,100,opt.labelStep):float()
if opt.dataset == 'scut-fbp' or opt.dataset == 'scut-fbp5500' or 
   opt.dataset == 'scut-fbp5500_1' or 
   opt.dataset == 'scut-fbp5500_2' or 
   opt.dataset == 'scut-fbp5500_3' or 
   opt.dataset == 'scut-fbp5500_4' or 
   opt.dataset == 'scut-fbp5500_5' then
   labelset = torch.range(1, 5, 0.1):float()
elseif opt.dataset == 'cfd' then
   labelset = torch.range(1, 7, 0.1):float()
elseif opt.dataset == 'hotnot' then
   labelset = torch.range(-3.6, 3.6, 0.1):float()
else
   labelset = torch.range(0, 100, 1):float()
end

if opt.loss == 'l1' or opt.loss == 'l2' then
   outDim  = 1
elseif opt.loss == 'rankbce' or opt.loss == 'rankmse' then
   outDim  = labelset:size(1) -1
elseif opt.loss == 'sm' or opt.loss == 'ldkl' or 'ldklexpl1' or 'ldklexpsmoothl1'then
   outDim = labelset:size(1)
end


modelPath = paths.concat(rootPath, 'model_60.t7')
assert(paths.filep(modelPath), 'File not found: ' .. modelPath)
print('Loading model from file: ' .. modelPath)
net = torch.load(modelPath):type(opt.tensorType)
--print(net)
-- Data loading
--trainLoader, valLoader = DataLoader.create(opt)
dataPath = paths.concat(opt.gen, opt.dataset .. '.t7')
data = torch.load(dataPath)
--print(data)
-- forward
timer = torch.Timer()
dataTimer = torch.Timer()
dataTime = dataTimer:time().real

nCrops =1
meanstd = {
   mean = { 0.5958, 0.4637, 0.4065 },
   std = {  0.2693, 0.2409, 0.2352 },
}  

result = {}
if opt.loss == 'ldklexpl1' or opt.loss == 'ldklexpl2'or 'ldklexpsmoothl1' then
   result.score = {torch.FloatTensor(data.val.imagePath:size(1), outDim),
                   torch.FloatTensor(data.val.imagePath:size(1), outDim),} 
   result.scoreb = {torch.FloatTensor(data.val.imagePath:size(1), 1),
                   torch.FloatTensor(data.val.imagePath:size(1), 1),} 
else
   result.score = {torch.FloatTensor(data.val.imagePath:size(1), outDim),
                torch.FloatTensor(data.val.imagePath:size(1), outDim),}
end
result.runTime = {torch.FloatTensor(data.val.imagePath:size(1)):fill(0),
               torch.FloatTensor(data.val.imagePath:size(1)):fill(0),} 
result.pred = {torch.FloatTensor(data.val.imagePath:size(1)):fill(0),
               torch.FloatTensor(data.val.imagePath:size(1)):fill(0),}

trans = {}
trans[1] = t.Compose{
         t.Scale(224),
         t.CenterCrop(224),
         t.ColorNormalize(meanstd),}

trans[2] = t.Compose{
         t.Scale(224),
         t.CenterCrop(224),
         t.HorizontalFlip(1),
         t.ColorNormalize(meanstd),}

net:evaluate()
for iter =  1, 2 do
   if iter == 1 then
      print('original prediction')
   elseif iter ==2 then
      print('flip prefiction')
   end
   local te = trans[iter]
   for i = 1, data.val.imagePath:size(1) do
       imgpath = paths.concat(data.basedir, ffi.string(data.val.imagePath[i]:data())) 
       img = image.load(imgpath, 'float')
       img2 = te(img):cuda()
       
       timer = torch.Timer()
       outs = net:forward(img2)
       if i%1000 ==0 then
           print((' |image: %d, Time %.3f '):
               format(i, timer:time().real)) 
       end 
       timer:reset()
       result.runTime[iter][i] = timer:time().real
       if opt.loss == 'ldklexpl1' or  opt.loss == 'ldklexpl2' or 'ldklexpsmoothl1'then
          local out1, out2 = unpack(outs)
          result.score[iter][i], result.scoreb[iter][i] = out1:float(), out2:float(2)
       else
          result.score[iter][i] = outs:float()
       end
      
    end
end

-- evaluation
class = data.val.imageClass:clone()
sigma = data.val.imageSigma:clone()
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
   elseif opt.loss == 'ldklexpl1' or opt.loss == 'ldklexpl2' or 'ldklexpsmoothl1'then
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

