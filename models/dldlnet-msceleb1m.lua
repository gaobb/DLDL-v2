local nn = require 'nn'
require 'cunn'
require 'cudnn'

local Avg = nn.SpatialAveragePooling
local Max = nn.SpatialMaxPooling
local function createModel(opt)
      featDim = 256
      outDim = opt.labelSet:size(1)
      model_path = "./MSCelebModels/"
      if opt.CR == 0.5 then
            featDim = 256
            print('loading model from ', model_path, "ThinMSCeleb1mNet-60.t7")
            model = torch.load(model_path.. "ThinMSCeleb1mNet-60.t7")
            model:remove(48)
            model:remove(47)

      elseif opt.CR == 0.25 then
            featDim = 128
            print('loading model from ', model_path, "TinyMSCeleb1mNet-45.t7")--model_45.t7
            model = torch.load(model_path.. "TinyMSCeleb1mNet-45.t7")
            model:remove(48)
            model:remove(47)
      end
   
   if opt.loss == 'ldkl' or  opt.loss == 'sm' then
      model:add(nn.Linear(featDim, outDim))
      model:add(nn.SoftMax())
   elseif opt.loss == 'rankbce' or opt.loss == 'rankmse' then
      model:add(nn.Linear(featDim, outDim-1))
      model:add(nn.Sigmoid())
   elseif opt.loss == 'l1' or opt.loss == 'l2' then
      model:add(nn.Linear(featDim, 1))
      model:add(nn.Tanh())
   elseif opt.loss == 'expl1' or opt.loss == 'expl2' then
      model:add(nn.Linear(featDim, outDim))
      model:add(nn.SoftMax())
      model:add(nn.ExpOut(opt.labelSet))
   end
   for k,v in pairs(model:findModules('nn.Linear')) do
      --v.bias = nil
      --v.gradBias = nil
      v.bias:zero()
   end
   model:type(opt.tensorType)
   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end
   print(model)
   model:get(1).gradInput = nil
   
   return model
end
return createModel
