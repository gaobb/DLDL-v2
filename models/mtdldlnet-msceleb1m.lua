local nn = require 'nn'
require 'cunn'
--require 'nnex'
require 'cudnn'
--require 'torch'
local Avg = nn.SpatialAveragePooling
local Max = nn.SpatialMaxPooling
local function createModel(opt)
     featDim = 256
     outDim = opt.labelSet:size(1)

     model_path = './Training-Models/'..opt.dataset..'/ldkl-hp-agenet-msceleb1m-CR'..opt.CR..'-Aug'..opt.dataAug..'-Step'..opt.labelStep..'/model_60.t7'
     if opt.CR == 0.5 then
            featDim = 256
     elseif opt.CR == 0.25 then
            featDim = 128
     end
     
     print('loading model from ', model_path)
     model = torch.load(model_path)

   if opt.loss == 'ldklexpl1' or opt.loss == 'ldklexpsmoothl1' or  opt.loss == 'ldklexpl2' then
      local joint = nn.ConcatTable()
      joint:add(nn.Identity()):add(nn.ExpOut(opt.labelSet))
      model:add(joint)
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
