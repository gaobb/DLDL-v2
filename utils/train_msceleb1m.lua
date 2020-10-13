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
require 'bbnn'
require 'torchx'
--require 'nnlr'
local matio = require 'matio'
local optim = require 'optim'

local M = {}
local Trainer = torch.class('train_msceleb1m.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   --self.Criterion = nn.BCECriterion():type(opt.tensorType)     
   --self.Criterion = nn.CrossEntropyCriterion():float():type(opt.tensorType)   
   self.Criterion = nn.ClassNLLCriterion():float():type(opt.tensorType)    --2017-10-09
   --ADAM
   self.optimState = optimState or {
      learningRate = opt.LR,     
      learningRateDecay = 0.0,
      beta1 = 0.9,
      beta2 = 0.999,
      epsilon = 10e-8,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   -- ADAM

   --[[if self.opt.netType == 'vggfacenet2' then
       local baseLR = self:learningRate(epoch)
       local baseWD = 0.0
       local lrs, wds = self.model:getOptimConfig(baseLR, baseWD)
       self.optimState.learningRates = lrs
       self.optimState.weightDecays = wds
       self.optimState.learningRate = baseLR 
       print('nnlr')
   else ]]
       self.optimState.learningRate = self:learningRate(epoch)
   --end

   
   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local dataTime = dataTimer:time().real
   local trainSize = dataloader:totalSize()
   local epochSize = dataloader:size()
   local lossSum = 0.0
   local N = 0
  
   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      -- local dataTime = dataTimer:time().real
      
      -- Copy input and target to the GPU
      self:copyInputs(sample)
      local loss
      
      local batchSize = self.input:size(1)
      local score = self.model:forward(self.input)
     
      --local gradInput
      feval = function (x)
           self.model:zeroGradParameters()
           -- forward
           --print(score:type(), score:size())
           --print(self.target:type(), self.target:size())
           loss = self.Criterion:forward(score, self.target:cuda())    
           -- backward
           self.Criterion:backward(score, self.target:cuda())

           local Grad = self.Criterion.gradInput
           self.model:backward(self.input, Grad)
           return  loss, self.gradParams 
      end
      --ADAM
      optim.adam(feval, self.params, self.optimState)
      --SGD
      --optim.sgd(feval, self.params, self.optimState)
     
      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      -- check that the storage didn't get changed due to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())
      if n%20==1 or n ==epochSize then
         print((' | Epoch: [%d %d/%d]  Err: %.7f, lr %.7f'):
               format(epoch, n, epochSize, loss, self.optimState.learningRate))  
      end
      --timer:reset()
      --dataTimer:reset()
   end 
    -- log 
    print((' | Epoch: [%d]  Time %.3f  Data %.3f Loss: %.7f lr %.7f'):
             format(epoch, timer:time().real, dataTime, lossSum / N,
                    self.optimState.learningRate)) 
   timer:reset()
   dataTimer:reset() 
   return lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set
   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local testSize = dataloader:totalSize()
   local epochSize = dataloader:size()
   
 
   local nCrops = self.opt.tenCrop and 10 or 1
   local accSum, lossSum = 0.0, 0.0
   local N = 0
   
   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
      local batchSize = self.input:size(1) / nCrops
      -- save info
      local score = self.model:forward(self.input)
      -- convert age to label distribution
      -- loss
      local loss = self.Criterion:forward(score, self.target:cuda())
      local top1, top5 = self:computeScore(score:float(), self.target:float(), nCrops)

      accSum  = accSum  + top1*batchSize
      lossSum = lossSum + loss*batchSize
      N = N + batchSize
      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   print((' * Finished epoch # %d  acc: %7.3f \n'):format(epoch, accSum / N))

   return accSum / N, lossSum / N
end

function Trainer:computeAcc(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)
   local pred = output:float():ge(0.5):int()
   -- Find which predictions match the target
   local correct = pred:eq(target:int())

   -- Top-1 score
   local acc = (correct:view(-1):sum() / (batchSize*40))

   return acc*100
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


function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'morph' or
          self.opt.dataset == 'chalearn16'  or
          self.opt.dataset == 'chalearn15'  or
          self.opt.dataset == 'chalearn16_vggface'  or
          self.opt.dataset == 'chalearn15_vggface'  or
          self.opt.dataset == 'google'  or
          self.opt.dataset == 'celeba'  or
          self.opt.dataset == 'msceleb1m'  or
          self.opt.dataset == 'cacd'  then
      decay = math.floor((epoch - 1) / 30) --20
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   elseif self.opt.dataset == 'cifar100' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(0.1, decay)
end

function Trainer:rampup(epoch, num_epochs)
   --local  = self.opt.nEpochs
   local rampup_length = 30.0
   local rp = 1.0
   if epoch <= rampup_length then
        local p = math.max(0.0, epoch-1) / rampup_length
        local p = 1.0 - p
        rp =  math.exp(-p*p*5.0)
   end
   return rp
end

function Trainer:weightup(epoch)
   local wt_max = self.opt.wt_max
   local num_epochs = self.opt.nEpochs
   local wt = 0.0
   if epoch > 1 then     
      wt = Trainer:rampup(epoch) * wt_max 
   end
   return wt
end

return M.Trainer
