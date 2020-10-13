--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--
require 'nn'
require 'torchx'
--local matio = require 'matio'
local optim = require 'optim'

local M = {}
local Trainer = torch.class('train_mtvgg.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   if opt.loss == 'ldklexpl1' then
      self.Criterion1 = nn.KLDivCriterion(true):float()     --klloss
      self.Criterion2 = nn.AbsCriterion():float()           --l1
   elseif opt.loss == 'ldklexpsmoothl1' then
      self.Criterion1 = nn.KLDivCriterion(true):float()     --klloss
      self.Criterion2 = nn.SmoothL1Criterion():float()     --smoothl1
   elseif opt.loss == 'ldklexpl2'  then
      self.Criterion1 = nn.KLDivCriterion(true):float()     --klloss
      self.Criterion2 = nn.MSECriterion():float()           --l1
   end
   --ADAM
   self.labelSet = torch.range(0, 100, opt.labelStep)
   self.optimState = optimState or {
      learningRate = opt.LR,     
      learningRateDecay = 0.0,
      beta1 = 0.9,
      beta2 = 0.999,
      epsilon = 10e-8,
   }
   self.labelSet = opt.labelSet
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   -- ADAM
   self.optimState.learningRate = self:learningRate(epoch)
   
   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local dataTime = dataTimer:time().real
   local trainSize = dataloader:totalSize()
   local epochSize = dataloader:size()
   local nCrops = self.opt.tenCrop and 10 or 1
   local maxMaeSum1, expMaeSum1, maxMaeSum2, expMaeSum2, maxMaeSumf, expMaeSumf, lossSum =  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
   local N = 0
   
  
   print('=> Training epoch # ' .. epoch)
   print(self.opt.lambda)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      -- Copy input and target to the GPU
      self:copyInputs(sample)
      local loss, loss1, loss2
      
      local batchSize = self.input:size(1)
      
      local outputs = self.model:forward(self.input)
      
      local score1, score2 = unpack(outputs) 
      local score1, score2 = score1:float(), score2:float() 
     
      -- convert age to label distribution
      local Target1, Target2
      
      if self.opt.loss == 'ldklexpl1' or self.opt.loss == 'ldklexpsmoothl1'or self.opt.loss == 'ldklexpl2' then
          Target1 = self:ageEncode(self.target, self.sigma, 'ldkl'):float()
          Target2 = self:ageEncode(self.target, self.sigma, 'expl1'):float()
      end
 
      --local gradInput
      feval = function (x)
           self.model:zeroGradParameters()
           -- forward
           
           loss1 = self.Criterion1:forward(score1, Target1)
           loss2 = self.Criterion2:forward(score2, Target2)
        
           -- backward
           self.Criterion1:backward(score1, Target1)
           self.Criterion2:backward(score2, Target2)
          
           loss = loss1+ loss2*self.opt.lambda   
           local Grad1 = self.Criterion1.gradInput
           local Grad2 = self.Criterion2.gradInput
           self.model:backward(self.input, {Grad1:cuda(), Grad2:mul(self.opt.lambda):cuda()}) 
          
           return  loss, self.gradParams 
      end
      --ADAM
      optim.adam(feval, self.params, self.optimState)
      
      if self.opt.loss == 'ldklexpl1' or self.opt.loss == 'ldklexpsmoothl1' or self.opt.loss == 'ldklexpl2' then
         maxMae1, expMae1 = self:computeMAE(score1, self.target, 'ldkl',   nCrops)
         maxMae2, expMae2 = self:computeMAE(score2, self.target, 'expl1', nCrops)
      end

      maxMaeSum1 = maxMaeSum1 + maxMae1*batchSize
      expMaeSum1 = expMaeSum1 + expMae1*batchSize

      maxMaeSum2 = maxMaeSum2 + maxMae2*batchSize
      expMaeSum2 = expMaeSum2 + expMae2*batchSize

      maxMaeSumf = maxMaeSumf + (maxMae1 + maxMae2 )/2*batchSize
      expMaeSumf = expMaeSumf + (expMae1 + expMae2 )/2*batchSize

      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      -- check that the storage didn't get changed due to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())
      if n%20==0 or n ==epochSize then
         print((' | Epoch: [%d %d/%d]  Err: %.7f= %.7f + %.7f, lr %.7f'):
               format(epoch, n, epochSize, loss, loss1, loss2, self.optimState.learningRate))  
      end
      --timer:reset()
      --dataTimer:reset()
   end 
    -- log 
    print((' | Epoch: [%d]  Time %.3f  Data %.3f expMae1: %7.3f expMae2: %7.3f, Loss: %.7f, lr %.7f'):
             format(epoch, timer:time().real, dataTime, expMaeSum1 / N, expMaeSum2 / N, lossSum / N,
                    self.optimState.learningRate)) 
   timer:reset()
   dataTimer:reset() 
   return maxMaeSumf / N, expMaeSumf / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set
   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local testSize = dataloader:totalSize()
   local epochSize = dataloader:size()
   
   local minage, maxage = 0, 100
   local nCrops = self.opt.tenCrop and 10 or 1
   local maxMaeSum1, expMaeSum1, maxMaeSum2, expMaeSum2, maxMaeSumf, expMaeSumf, lossSum =  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
   local N = 0
  
   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
      local batchSize = self.input:size(1) / nCrops
      -- save info
      local outputs = self.model:forward(self.input)

      local score1, score2 = unpack(outputs)
      local score1, score2 = score1:float(), score2:float()--, score3:float()
     
      local Target1, Target2
      
      if self.opt.loss == 'ldklexpl1' or self.opt.loss == 'ldklexpsmoothl1' or self.opt.loss == 'ldklexpl2' then
          Target1 = self:ageEncode(self.target, self.sigma, 'ldkl'):float()
          Target2 = self:ageEncode(self.target, self.sigma, 'expl1'):float()
      end

      local loss, loss1, loss2
      -- loss
      loss1 = self.Criterion1:forward(score1, Target1)
      loss2 = self.Criterion2:forward(score2, Target2)
 
      loss = loss1+ loss2*self.opt.lambda
     
      if self.opt.loss == 'ldklexpl1' or self.opt.loss == 'ldklexpsmoothl1' or self.opt.loss == 'ldklexpl2' then
         maxMae1, expMae1 = self:computeMAE(score1, self.target, 'ldkl',   nCrops)
         maxMae2, expMae2 = self:computeMAE(score2, self.target, 'expl1', nCrops)
      end

      maxMaeSum1 = maxMaeSum1 + maxMae1*batchSize
      expMaeSum1 = expMaeSum1 + expMae1*batchSize

      maxMaeSum2 = maxMaeSum2 + maxMae2*batchSize
      expMaeSum2 = expMaeSum2 + expMae2*batchSize

      maxMaeSumf = maxMaeSumf + (maxMae1 + maxMae2 )/2*batchSize
      expMaeSumf = expMaeSumf + (expMae1 + expMae2 )/2*batchSize

      lossSum = lossSum + loss*batchSize
      N = N + batchSize
      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   print((' * Finished epoch # %d  expMae1: %7.3f expMae2: %7.3f \n'):format(epoch, expMaeSum1 / N, expMaeSum2 / N))

   return maxMaeSumf / N, expMaeSumf / N, lossSum / N--, epochInfos
end

function Trainer:computeMAE(output, target, loss, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end
   local minage, maxage = self.labelSet:min(), self.labelSet:max()
   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)
   local maxPred, expPred
   if loss == 'ldkl' or loss == 'sm' then
      local _ , maxind = output:float():topk(1, 2, true, true) -- descending
      maxPred = self.labelSet:index(1, maxind:squeeze())
      expPred = output*self.labelSet:squeeze()
   elseif loss == 'expl1' then
      expPred = output:squeeze()
      maxPred = expPred
   elseif loss == 'l1' then
      expPred = Trainer:mapminmax(output, {-1,1}, {minage, maxage}):squeeze()
      maxPred = expPred
   elseif loss == 'cdfl2' then
      local sumcount = (output:float():le(0.5):sum(2)):float()
      expPred = self.labelSet:index(1, sumcount:long():squeeze())
      maxPred = expPred
   elseif loss == 'rank' then
       local sumcount = (output:float():ge(0.5):sum(2)):float() + 1
      expPred = self.labelSet:index(1, sumcount:long():squeeze())
      maxPred = expPred
   end
   
   -- Find which predictions match the target
   local maxMae = (maxPred:float() - target:float()):abs():sum() / batchSize
   local expMae = (expPred:float() - target:float()):abs():sum() / batchSize
   return maxMae, expMae
end

function Trainer:computeScore(output, target, loss, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():topk(5, 2, true, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(predictions))

   -- Top-1 score
   local top1 = (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

local function getCudaTensorType(tensorType)
  if tensorType == 'torch.CudaHalfTensor' then
     return cutorch.createCudaHostHalfTensor()
  elseif tensorType == 'torch.CudaDoubleTensor' then
    return cutorch.createCudaHostDoubleTensor()
  else
     return cutorch.createCudaHostTensor()
  end
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
  
   self.input = self.input or (self.opt.nGPU == 1
      and torch[self.opt.tensorType:match('torch.(%a+)')]()
      or getCudaTensorType(self.opt.tensorType))
   self.target = self.target or  (self.opt.nGPU == 1
      and torch[self.opt.tensorType:match('torch.(%a+)')]()
      or getCudaTensorType(self.opt.tensorType))
   self.sigma = self.sigma or  (self.opt.nGPU == 1
      and torch[self.opt.tensorType:match('torch.(%a+)')]()
      or getCudaTensorType(self.opt.tensorType))

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
   self.sigma:resize(sample.sigma:size()):copy(sample.sigma)
end

function Trainer:ageEncode(age, sigma, loss)
    local target
    local minlabel, maxlabel = self.labelSet:min(), self.labelSet:max()
    if loss == 'ldkl' then
         target  = Trainer:genld(age, sigma, self.labelSet)
    elseif loss == 'expl1' then
         target  = age
    elseif loss == 'sm' then
         local labelLen = self.labelSet:size(1)
         local labelNum = age:size(1)
         local _, minind = (age:reshape(labelNum, 1):expand(labelNum, labelLen) - self.labelSet:reshape(1, labelLen):expand(labelNum, labelLen)):abs():min(2)
         target = minind:float():squeeze()
    elseif loss == 'l1' or loss == 'smoothl1' or loss == 'l2' then
         target  = Trainer:mapminmax(age, {minlabel, maxlabel}, {-1,1})
    end
    return target
end


function Trainer:ageRank(age, sigma, labelset)
    --local roundAge = torch.round(age)
    local minlabel, maxlabel = labelset:min(), labelset:max()
    local labelLen = labelset:size(1)
    local labelNum = age:size(1)
    local RankTemp = (age:reshape(labelNum, 1):expand(labelNum, labelLen) - labelset:reshape(1, labelLen):expand(labelNum, labelLen)):gt(0):float()
    local rankCode = RankTemp:index(2, torch.range(1,labelLen-1):long())
    return rankCode
end

function Trainer:gencdf(age, sigma, labelset)
    local minlabel, maxlabel = labelset:min(), labelset:max()
    local labelLen = labelset:size(1)
    local labelNum = age:size(1)
    local RankTemp = (age:reshape(labelNum, 1):expand(labelNum, labelLen) - labelset:reshape(1, labelLen):expand(labelNum, labelLen)):lt(0):float()
    local rankCode = RankTemp:index(2, torch.range(1,labelLen):long())
    return rankCode
end

function Trainer:ageSRank(age, sigma, labelset)
    local labelLen = labelset:size(1)
    local ageRank = torch.Tensor(age:size(1), labelLen-1):fill(1)
     for i = 1, age:size(1) do
         if sigma[i] ==0 then
            sigma[i] = sigma[i] + 1e-10
         end
         local pdf = torch.pow((labelset - age[i])/sigma[i], 2):mul(-0.5):exp()
         local pdf = pdf/pdf:sum()
         local ncdf = 1 -torch.cumsum(pdf)
         ageRank[i] = ncdf[{{1,-2}}]
     end
    return ageRank
end



function Trainer:genld(age, sigma, labelset)
  --local labelStep = 100/(labelset:size(1)-1)
  --print(labelStep)
  local age, sigma = age:squeeze(), sigma:squeeze()--:mul(labelStep)
  --sigma = sigma * self.opt.labelStep   --2017-10-29
  local minlabel, maxlabel = labelset:min(), labelset:max()
  local labelLen = labelset:size(1)
  local lds = torch.FloatTensor(age:size(1),  labelLen)
  for i = 1, age:size(1) do
     if sigma[i] ==0 then
        sigma[i] = sigma[i] + 1e-10
     end
     local ld = torch.pow((labelset - age[i])/sigma[i], 2):mul(-0.5):exp()
     lds[i] = ld/ld:sum()
  end
  return lds
end

function Trainer:genmaxld(age, sigma, labelset)
  age, sigma = age:squeeze(), sigma:squeeze()
  local minlabel, maxlabel = labelset:min(), labelset:max()
  local labelLen = labelset:size(1)
  local lds = torch.FloatTensor(age:size(1),  labelLen)
  for i = 1, age:size(1) do
     if sigma[i] ==0 then
        sigma[i] = sigma[i] + 1e-10
     end
     local ld = torch.pow((labelset  - age[i])/sigma[i], 2):mul(-0.5):exp()
     lds[i] = ld/ld:max()
  end
  return lds
end


function Trainer:mapminmax(x, o, t)
   local f = (t[2]-t[1])/(o[2]-o[1])
   return ((x - o[1]):mul(f) + t[1])
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = math.floor((epoch - 1) / 30) --20
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
