CUDA_VISIBLE_DEVICES=9 th


require 'nn'
require 'cunn'
require 'cudnn'
require 'torchx'
ffi = require 'ffi'
require 'image'
t = require 'datasets/transforms'
matio = require 'matio'


-- Model loading
opt = {}
opt.dataset = 'chalearn16'
opt.loss = 'ldklcdfl2' 
opt.dataAug = 'true'
opt.lambda =1
opt.gen = '/mnt/data3/gaobb/projects/DLDL-v2/gen'
rootPath = '/mnt/data3/gaobb/projects/DLDL-v2/'
modelPath = paths.concat(rootPath,opt.dataset,'max2_avg7',opt.loss.."_thinvggbn_maxavg_msceleb1m_mt_CR0.5_Aug_Lambda_"..opt.lambda, 'model_60.t7')
opt.labelStep = 1
labelset = torch.range(0, 100, opt.labelStep):float()
outDim  = 101

assert(paths.filep(modelPath), 'File not found: ' .. modelPath)
print('Loading model from file: ' .. modelPath)
net = torch.load(modelPath):cuda()
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


pdfscore = {torch.FloatTensor(data.val.imagePath:size(1), outDim),
            torch.FloatTensor(data.val.imagePath:size(1), outDim),} 
cdfscore = {torch.FloatTensor(data.val.imagePath:size(1), outDim),
            torch.FloatTensor(data.val.imagePath:size(1), outDim),} 


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
       local out1, out2 = unpack(outs)
       pdfscore[iter][i], cdfscore[iter][i] = out1:float(), out2:float(2)
    end
end

-- evaluation
class = data.val.imageClass:clone()
sigma = data.val.imageSigma:clone()
inds = torch.LongTensor(torch.find(sigma, 0))
if inds:dim() >=1 then
   sigma:indexFill(1, inds:long(), 1e-10)
end

minl, maxl = labelset:min(), labelset:max()
pdfpred1  = pdfscore[1]:float()*(labelset)
pdfpred2  = pdfscore[2]:float()*(labelset)

pdfpred = (pdfpred1 + pdfpred2)/2
mae = (pdfpred-class):abs():mean()


matio = require 'matio'
matio.save('ldkl')

sumcount = (cdfscore[1]:float():le(0.5):sum(2)):float()+1
cdfpred1  = labelset:index(1, sumcount:long():squeeze())
sumcount = (cdfscore[2]:float():le(0.5):sum(2)):float()+1
cdfpred2  = labelset:index(1, sumcount:long():squeeze())
cdfpred = (cdfpred1 + cdfpred1)/2
mae = (cdfpred-class):abs():mean()

mae = ((pdfpred + cdfpred)/2-class):abs():mean()


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
