CUDA_VISIBLE_DEVICES=1 th 
require 'nn'
require 'cunn'
require 'cudnn'

ffi = require 'ffi'
require 'image'
t = require 'datasets/transforms'

matio = require 'matio'
DataLoader = require 'dataloader'
opt = {}
opt.gen = "/mnt/data3/gaobb/projects/Super_Age/gen"
opt.tensorType = 'torch.CudaTensor'
opt.manualSeed = 0
opt.dataAug = false
opt.batchSize = 128
opt.nThreads = 10
-- Model loading
rootPath = "/mnt/data3/gaobb/projects/Super_Age/msceleb1m/CE_thinvggbn_avg_Aug_cr0"
rootPath = "/mnt/data3/gaobb/projects/Super_Age/msceleb1m/CE_thinvggbn_avg_Aug_cr0.5"
rootPath = "/mnt/data3/gaobb/projects/Super_Age/msceleb1m/CE_thinvggbn_avg_Aug_cr1"
rootPath = "/mnt/data3/gaobb/projects/Super_Age/msceleb1m/CE_thinvggbn_avg_Aug_cr0.25"
opt.dataset = 'msceleb1m'

modelPath = paths.concat(rootPath, 'model_60.t7')
assert(paths.filep(modelPath), 'File not found: ' .. modelPath)
print('Loading model from file: ' .. modelPath)
net = torch.load(modelPath)
net:remove(48)
net:add(nn.SoftMax())

net:type(opt.tensorType)
-- Data loading
--trainLoader, valLoader = DataLoader.create(opt)
dataPath = paths.concat(opt.gen, opt.dataset .. '.t7')
data = torch.load(dataPath)


-- forward
timer = torch.Timer()
dataTimer = torch.Timer()
dataTime = dataTimer:time().real
out_dim  = 54073
nCrops =1
meanstd = {
   mean = { 0.5958, 0.4637, 0.4065 },
   std = {  0.2693, 0.2409, 0.2352 },
}  

te = t.Compose{
         t.Scale(224),
         t.CenterCrop(224),
         --t.HorizontalFlip(1),
         t.ColorNormalize(meanstd),
      }

maxPred = torch.FloatTensor(data.val.imagePath:size(1)):fill(0)
runTime = torch.FloatTensor(data.val.imagePath:size(1)):fill(0)
score = torch.FloatTensor(data.val.imagePath:size(1), out_dim)
net:evaluate()

for i = 1, data.val.imagePath:size(1) do
   imgpath = paths.concat(data.basedir, ffi.string(data.val.imagePath[i]:data())) 
   img = image.load(imgpath, 'float')
   img2 = te(img)
   input = img2:cuda()
   timer = torch.Timer()
   out = net:forward(input)
   runTime[i] = timer:time().real 
   if i %10000 == 0 then
      print((' |image: %d, Time %.3f '):format(i, timer:time().real)) 
   end
   score[i] = out:float()
   timer:reset()
end


_ , maxv = score:float():topk(1, 2, true, true) -- descending
Pred = maxv:float():squeeze() 


-- evaluation
class = data.val.imageClass
Acc = Pred:eq(class):sum()/data.val.imagePath:size(1)
print(('|Acc %.4f, Avg Time: %.4f s'):format(Acc*100, runTime:mean()))


|Acc 54.6289, Avg Time: 0.0121 s CR 0    60th-epoch

|Acc 48.0776, Avg Time: 0.0051 s CR 0.25 11th-epoch
|Acc 49.9057, Avg Time: 0.0108 s CR 0.25 15th-epoch
|Acc 53.2872, Avg Time: 0.0073 s CR 0.25 35th-epoch
|Acc 53.3936, Avg Time: 0.0026 s         44th-epoch
|Acc 53.363,  Avg Time: 0.0073 s CR 0.25 45th-epoch
|Acc 53.4527, Avg Time: 0.0024 s CR 0.25 45th-epoch

|Acc 57.7469, Avg Time: 0.0126 s CR 0.5  60th-epoch

|Acc 53.5970, Avg Time: 0.0261 s CR 1   5th-epoch
|Acc 55.7025, Avg Time: 0.0267 s        7th-epoch
|Acc 56.4154, Avg Time: 0.0238 s        9th-epoch
|Acc 57.8024, Avg Time: 0.0165 s        17th-epoch
|Acc 58.0197, Avg Time: 0.0156 s        19th-epoch
|Acc 58.2943, Avg Time: 0.0293 s        22th-epoch
|Acc 58.6365, Avg Time: 0.0156 s	27th-epoch
|Acc 60.2362, Avg Time: 0.0244 s	31th-epoch
|Acc 60.3000, Avg Time: 0.0347 s        35th-epoch


CUDA_VISIBLE_DEVICES=14 th
require 'nn'
require 'cudnn'
net = torch.load('model_10.t7')
net:remove(48)
net:remove(47)
pa, gradpa = net:getParameters()
print(pa:size(1)*4/1024/1024)


for l = 1, 49 do
    if net:get(l).gradWeight ~= nil then
       net:get(l).gradWeight = nil
    end
    if net:get(l).gradBias ~= nil then
       net:get(l).gradBias = nil
    end
end
torch.save('net.t7', net)
 cudnn.convert(net, cudnn)
 input = torch.CudaTensor(128, 3, 224,224):normal(0,1)
 timer = torch.Timer()
 score = net:forward(input):float()
 runTime = timer:time().real 
 print((' Time %.3f '):format(runTime)) 
 timer:reset()

 cudnn.convert(net, nn)
 input = torch.CudaTensor(128, 3, 224,224):normal(0,1)
 timer = torch.Timer()
 score = net:forward(input):float()
 runTime = timer:time().real 
 print((' Time %.3f '):format(runTime)) 
 timer:reset()
data
{
  basedir : "/mnt/data3/gaobb/image_data/image_faces/age_faces"
  train : 
    {
      imageId : IntTensor - size: 2476
      imagePath : CharTensor - size: 2476x46
      feats : FloatTensor - size: 2476x256
      imageSigma : FloatTensor - size: 2476
      imageMask : ByteTensor - size: 2476
      imageClass : FloatTensor - size: 2476
    }
  val : 
    {
      imageId : IntTensor - size: 1136
      imagePath : CharTensor - size: 1136x46
      feats : FloatTensor - size: 1136x256
      imageSigma : FloatTensor - size: 1136
      imageMask : ByteTensor - size: 1136
      imageClass : FloatTensor - size: 1136
    }
}
data.train.imagePath = data.train.imagePath:int()
data.val.imagePath = data.val.imagePath:int()
matio.save('chalearn16_feats.mat', data)

