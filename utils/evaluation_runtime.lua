CUDA_VISIBLE_DEVICES=2 th

require 'nn'
require 'image'
require 'cudnn'
require 'loadcaffe'
cudnn.benchmark = true
cudnn.verbose = false
-- vggface
vggfaceProb = '/home/gaobb/Projects/vgg_face_caffe/VGG_FACE_deploy.prototxt'
vggfacePath = '/home/gaobb/Projects/vgg_face_caffe/VGG_FACE.caffemodel'
net = loadcaffe.load(vggfaceProb, vggfacePath)
net:remove(39)
net:add(nn.Linear(4096,101))
net:add(nn.SoftMax())


cudnn.convert(net, cudnn)
net:cuda()
net:evaluate()

steps = 20
input = torch.FloatTensor(32,3,224,224):normal(0,1):cuda()
sys.tic()
for t = 1, steps do
    output = net:updateOutput(input)
end
cutorch.synchronize()
tmf = sys.toc()/steps
print('time : ', tmf*1000, 'ms')

deltaT = torch.FloatTensor(20):fill(0)
for i=1, 20 do
   local input = torch.randn(32, 3, 224, 224):type("torch.CudaTensor")
   cutorch.synchronize()
   timer = torch.Timer()
   net:forward(input)
   cutorch.synchronize()
   deltaT[i] = timer:time().real
   print("Forward time: " .. deltaT[i])
end
print("mean Forward time: " .. 1000*deltaT:mean().."ms")



CUDA_VISIBLE_DEVICES=12 th

require 'nn'
require 'image'
require 'loadcaffe'
require 'cudnn'
cudnn.benchmark = true
cudnn.verbose = false
-- resnet50.t7
netPath = './premodels/resnet-50.t7'
net = torch.load(netPath)
net:remove(11)
net:add(nn.Linear(2048,41))
net:add(nn.SoftMax())
cudnn.convert(net, cudnn)
net:cuda()
net:evaluate()

deltaT = torch.FloatTensor(20):fill(0)
for i=1, 20 do
   local input = torch.randn(32, 3, 224, 224):type("torch.CudaTensor")
   cutorch.synchronize()
   timer = torch.Timer()
   net:forward(input)
   cutorch.synchronize()
   deltaT[i] = timer:time().real
   print("Forward time: " .. deltaT[i])
end
print("mean Forward time: " .. 1000*deltaT:mean().."ms")


CUDA_VISIBLE_DEVICES=2 th
require 'nn'
require 'image'
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
cudnn.verbose = false
-- DLEP
netPath = './chalearn16/max2_avg7/ldklexpl1_thinvggbn_maxavg_msceleb1m_mt_CR0.5_Aug_Lamda_1/model_60.t7'
netPath = './chalearn16/max2_avg7/ldklexpl1_thinvggbn_maxavg_msceleb1m_mt_CR0.25_Aug_Lambda_1/model_60.t7'
netPath = './scut-fbp/max2_avg7/ldklexpl1_thinvggbn_maxavg_msceleb1m_mt_CR0.5_Aug_Lambda_1/model_60.t7'
netPath = './scut-fbp/max2_avg7/ldklexpl1_thinvggbn_maxavg_msceleb1m_mt_CR0.25_Aug_Lambda_1/model_60.t7'
net = torch.load(netPath)
net:remove(49)
cudnn.convert(net, cudnn)
net:cuda()
net:evaluate()

deltaT = torch.FloatTensor(20):fill(0)
for i=1, 20 do
   local input = torch.randn(32, 3, 224, 224):type("torch.CudaTensor")
   cutorch.synchronize()
   timer = torch.Timer()
   net:forward(input)
   cutorch.synchronize()
   deltaT[i] = timer:time().real
   print("Forward time: " .. deltaT[i])
end
print("mean Forward time: " .. 1000*deltaT:mean().."ms")




net:evaluate()
input = torch.CudaTensor(128,3,224,224):normal(0,1)
time = torch.FloatTensor(10):fill(0)
for i = 1,10 do
timer = torch.Timer()
out = net:forward(input)
time[i] = timer:time().real
timer:reset()
end
print('time : ', time:mean()*1000, 'ms')

-- conert kernels in first layer into RGB format instead of BGR,
-- which is the order in which it was trained in caffe
w = net:get(1).weight:clone()
net:get(1).weight[{{},1,{},{}}]:copy(w[{{},3,{},{}}])
net:get(1).weight[{{},3,{},{}}]:copy(w[{{},1,{},{}}])

-- add sofmax layer
net:add(nn.SoftMax())
net:float()
net:evaluate()

im = image.load('./ak.png',3,'float')
im = im*255
mean = {93.5940, 104.7624, 129.1863} --RGB order

for i=1,3 do im[i]:add(-mean[i]) end
prob = net:forward(im)
maxval,maxid = prob:max(1)
print(maxval, maxid)

-- remove last fc layers
for l = 40, 31, -1 do
    vgg16:remove(l)
end

