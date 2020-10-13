CUDA_VISIBLE_DEVICES=9 th
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
opt.sourcedata = 'scut-fbp'
opt.targetdata = 'cfd'

opt.sourcedata = 'cfd'
opt.targetdata = 'scut-fbp'

opt.sourcedata = 'morph'
opt.targetdata = 'chalearn16'
opt.sourcedata = 'chalearn16'
opt.targetdata = 'morph'
--opt.dataset = 'morph'
--opt.dataset = 'chalearn16'
opt.nThreads = 10
opt.loss = 'ldklexpl1'
-- Model loading
rootPath = paths.concat("/mnt/data3/gaobb/projects/DLDL-v2/",opt.sourcedata,'max2_avg7',opt.loss.."_thinvggbn_maxavg_msceleb1m_mt_CR0.5_Aug_Lamda_1")
local labelset
if opt.sourcedata == 'scut-fbp' then
   labelset = torch.range(1, 5, 0.1):float()
elseif opt.sourcedata == 'cfd' then
   labelset = torch.range(1, 7, 0.1):float()
else
   labelset = torch.range(0, 100, 1):float()
end
outDim = labelset:size(1)
opt.tensorType = 'torch.CudaTensor'
modelPath = paths.concat(rootPath, 'model_60.t7')
assert(paths.filep(modelPath), 'File not found: ' .. modelPath)
print('Loading model from file: ' .. modelPath)
net = torch.load(modelPath):type(opt.tensorType)
-- Data loading
--trainLoader, valLoader = DataLoader.create(opt)
dataPath = paths.concat(opt.gen, opt.targetdata .. '.t7')
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

class = data.val.imageClass:clone()
sigma = data.val.imageSigma:clone()
maes = torch.FloatTensor(dh:numel(),dw:numel())
for i = 1, data.val.imagePath:size(1) do
   imgpath = paths.concat(data.basedir, ffi.string(data.val.imagePath[i]:data())) 
   img = image.load(imgpath, 'float')
   fimg = image.hflip(img)
   img1 = te(img)
   fimg1 = te(fimg)
   imt = torch.FloatTensor(2,3,224,224)
   imt[{{1},{},{}}] = img1:clone()
   imt[{{2},{},{}}] = fimg1:clone()
   outs = net:forward(imt:cuda())
   local out1, out2 = unpack(outs)
   pred[i] =  out2:mean()
end
-- evaluation
maes = (pred-class):abs():mean()
matio.save(opt.sourcedata..opt.targetdata..'.mat', {pred=pred, class=class, sigma=sigma})

