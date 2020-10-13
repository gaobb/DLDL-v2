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
require 'torch'
require 'torchx'

local optim = require 'optim'

local M = {}
local Trainer = torch.class('train_net.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   if opt.loss == 'ldkl' then
      self.Criterion = nn.KLDivCriterion(true):float() 
   elseif opt.loss == 'sm' then
      --self.Criterion = nn.CrossEntropyLogCriterion():float() 
      self.Criterion = nn.ClassNLLCriterion():float()   
   elseif opt.loss == 'l1'  or opt.loss == 'expl1' then
      self.Criterion = nn.AbsCriterion(true):float()   
   elseif opt.loss == 'l2'  or opt.loss == 'expl2' then
      self.Criterion = nn.MSECriterion(true):float()         
   elseif opt.loss == 'rankbce' then
      self.Criterion = nn.BCECriterion():float()            -- bce loss
   elseif opt.loss == 'rankmse' then
      self.Criterion = nn.MSECriterion():float()            -- l2 loss
   end

   --ADAM
   self.optimState = optimState or {
      learningRate = opt.LR,     
      learningRateDecay = 0.0,
      beta1 = 0.9,
      beta2 = 0.999,
      epsilon = 10e-8,
   }
   if opt.loss == 'ldkl' or 
      opt.loss == 'sm' then
      self.outsize = opt.labelSet:size(1)
   elseif opt.loss == 'rankbce' or opt.loss == 'rankmse' then
      self.outsize = opt.labelSet:size(1)-1
   elseif opt.loss == 'l1' or  opt.loss == 'l2' or opt.loss == 'expl1' or  opt.loss == 'expl2' then 
      self.outsize = 1  
   end
   self.labelSet = opt.labelSet
   self.opt = opt
   print(opt)
   self.params, self.gradParams = model:getParameters()
end

function Trainer:center_loss(centers, features, target, alpha)
    local target = target - target:min() + 1
    local batch_size = target:size(1)
  
    local centers_batch = centers:index(1, target:long())
    local L2Criterion = nn.MSECriterion()
    local centerloss = L2Criterion:forward(features,  centers_batch)
    
    local diff = centers_batch - features
    for c = 1, centers:size(1) do
        local indx = torch.LongTensor(torch.find(target, c))
        if indx:dim() >=1 then
            centers[c] = centers[c] - diff:index(1, indx):mean(1) *alpha
        end
    end
    local graddiff = features - centers_batch
    return centerloss, centers, graddiff
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
   local maxMaeSum, expMaeSum, lossSum = 0.0, 0.0, 0.0
   local N = 0
   local nCrops = self.opt.tenCrop and 10 or 1
   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      -- Copy input and target to the GPU
      self:copyInputs(sample)
      local loss, centerloss
      
      local batchSize = self.input:size(1)
      
      local score = self.model:forward(self.input):float()
      -- age encoding
      local Target = self:ageEncode(self.target, self.sigma, self.opt.loss)
      
      feval = function (x)
           self.model:zeroGradParameters()
           -- forward
           loss = self.Criterion:forward(score, Target)  
           -- backward
           self.Criterion:backward(score, Target)
           
           local Grad = self.Criterion.gradInput
           self.model:backward(self.input,  Grad:cuda())
           return  loss, self.gradParams 
      end
      --ADAM
      optim.adam(feval, self.params, self.optimState)
      --SGD
      --optim.sgd(feval, self.params, self.optimState)
     
      lossSum = lossSum + loss*batchSize

     local maxMae, expMae = self:computeMAE(score, self.target, nCrops)
      maxMaeSum = maxMaeSum + maxMae*batchSize
      expMaeSum = expMaeSum + expMae*batchSize

      N = N + batchSize

      -- check that the storage didn't get changed due to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())
      if n%20==0 or n ==epochSize then
         print((' | Epoch: [%d %d/%d]  Err: %.7f, lr %.7f'):
               format(epoch, n, epochSize, loss,  self.optimState.learningRate))  
      end
   end 
    -- log 
   print((' | Epoch: [%d]  Time %.3f  Data %.3f Loss: %.7f lr %.7f maxMae: %7.3f expMae: %7.3f'):
             format(epoch, timer:time().real, dataTime, lossSum / N,
                    self.optimState.learningRate, maxMaeSum / N, expMaeSum / N)) 
   timer:reset()
   dataTimer:reset() 
   return maxMaeSum / N, expMaeSum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set
   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local testSize = dataloader:totalSize()
   local epochSize = dataloader:size()
  
   local nCrops = self.opt.tenCrop and 10 or 1
   local maxMaeSum, expMaeSum, lossSum = 0.0, 0.0, 0.0
   local N = 0
   
   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
      local batchSize = self.input:size(1) / nCrops
     
      local score = self.model:forward(self.input):float()
      -- convert age to label distribution
      local Target = self:ageEncode(self.target, self.sigma, self.opt.loss)
      -- loss
      local loss = self.Criterion:forward(score, Target)
      local maxMae, expMae = self:computeMAE(score, self.target, nCrops)
     
      maxMaeSum = maxMaeSum + maxMae*batchSize
      expMaeSum = expMaeSum + expMae*batchSize
      lossSum = lossSum + loss*batchSize
      N = N + batchSize
      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   print((' * Finished epoch # %d  maxMae: %7.3f expMae: %7.3f \n'):format(epoch, maxMaeSum / N, expMaeSum / N))

   return maxMaeSum / N, expMaeSum / N, lossSum / N
end

function Trainer:computeMAE(output, target, nCrops)
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
   if self.opt.loss == 'ldkl' or  
      self.opt.loss == 'expkl' or 
      self.opt.loss == 'sm' then
      local _ , maxind = output:float():topk(1, 2, true, true) -- descending
      maxPred = self.labelSet:index(1, maxind:squeeze())
      expPred = output*self.labelSet:squeeze()   
   elseif self.opt.loss == 'l1' or 
          self.opt.loss == 'l2'  then
      expPred = Trainer:mapminmax(output, {-1,1}, {minage, maxage}):squeeze()
      maxPred = expPred
   elseif self.opt.loss == 'expl1' or 
          self.opt.loss == 'expl2'  then
      expPred = output:squeeze()
      maxPred = expPred
   elseif self.opt.loss == 'rankbce' or 
          self.opt.loss == 'rankmse'  then
      local sumcount = (output:float():ge(0.5):sum(2)):float() + 1
      expPred = self.labelSet:index(1, sumcount:long():squeeze())
      maxPred = expPred
   end
   
   -- Find which predictions match the target
   local maxMae = (maxPred - target:float()):abs():sum() / batchSize
   local expMae = (expPred - target:float()):abs():sum() / batchSize
   return maxMae, expMae
end

function Trainer:computeScore(output, target, nCrops)
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
    if loss == 'ldkl' or loss == 'expkl' or loss == 'rekl'then
         target  = Trainer:genld(age, sigma, self.labelSet)
    elseif loss == 'sm' then
   
         local labelLen = self.labelSet:size(1)
         local labelNum = age:size(1)
         local _, minind = (age:reshape(labelNum, 1):expand(labelNum, labelLen) - self.labelSet:reshape(1, labelLen):expand(labelNum, labelLen)):abs():min(2)
         target = minind:float():squeeze()
    elseif loss == 'l1'  or loss == 'l2' then
         target  = Trainer:mapminmax(age, {minlabel, maxlabel}, {-1,1})
    elseif loss == 'expl1'  or loss == 'expl2' then
         target  = age
    elseif loss == 'rankbce' or loss == 'rankmse' then
         target = Trainer:ageRank(age, sigma, self.labelSet)
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
    --local ageRankTemp = torch.Tensor(labelLen, labelLen):fill(-1):triu():add(1):index(2, torch.range(1,labelLen-1):long())
    --local ageRank = ageRankTemp:index(1, (roundAge+1):long())
    --print('rankcode size ', rankCode:size())
    return rankCode
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
   local decay = 0
   decay = math.floor((epoch - 1) / 30) 
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
