local nn = require 'nn'
require 'cunn'
local SConvolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local BatchNorm = nn.BatchNormalization
local function createModel(opt)
   local modelType = 'D' -- on a titan black, B/D/E run out of memory even for batch-size 32

   -- Create tables describing VGG configurations A, B, D, E
   local cfg = {}
   if modelType == 'A' then
      cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'B' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'D' then
        --output size: 224->224->112->112->112->56->56->56->56->28->28->28->28->14->14->14->14->7
        if opt.CR == 1 then -- compression rate
           cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'}      --vgg16
        elseif opt.CR == 1/2 then  
           cfg = {32, 32, 'M', 64, 64,   'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'}      --1/2
        elseif opt.CR == 1/4  then
           cfg = {16, 16, 'M', 32, 32,   'M', 64,  64,  64,  'M', 128, 128, 128, 'M', 128, 128, 128, 'M'}      --1/4
        elseif opt.CR == 1/8  then
           cfg = {8,   8, 'M', 16, 16,   'M', 32,  32,  32,  'M', 64, 64, 64,    'M', 64, 64, 64,    'M'}      --1/8
        elseif opt.CR == 0 then
           cfg = {32, 32, 'M', 64, 64,   'M', 128, 128, 128, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'} 
        end 
   elseif modelType == 'E' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end

   local model = nn.Sequential()
   local iChannels, oChannels
   do
      iChannels = 3;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            model:add(Max(2,2,2,2))
         else
            oChannels = v;
            model:add(SConvolution(iChannels,oChannels,3,3,1,1,1,1))
            model:add(SBatchNorm(oChannels))
            model:add(ReLU(true))
            iChannels = oChannels;
         end
      end
   end
   
   model:add(Avg(7,7,1,1))
   model:add(nn.View(oChannels):setNumInputDims(3))
   if opt.dataset == 'celeba' then
         model:add(nn.Linear(oChannels, 40))
         model:add(nn.Sigmoid())
   elseif  opt.dataset == 'msceleb1m' then
         model:add(nn.Linear(oChannels, 54073))
         model:add(nn.LogSoftMax())
   end
   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end
   print(model)

   ConvInit('cudnn.SpatialConvolution')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end
   model:get(1).gradInput = nil
   
   return model
end


return createModel
