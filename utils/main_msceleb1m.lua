--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'gnuplot'
require 'torchx'
local matio = require 'matio'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train_msceleb1m'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

-- we don't  change this to the 'correct' type (e.g. HalfTensor), because math
-- isn't supported on that type.  Type conversion later will handle having
-- the correct type.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(5)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
--local checkpoint, optimState = checkpoints.latest(opt)

-- Create model

local model, criterion = models.setup(opt, checkpoint)
local checkpoint = {}
checkpoint = {epoch = 34,
              modelFile = 'model_34.t7',
              optimFile = 'optimState_34.t7',}
local optimPath = paths.concat(opt.save, checkpoint.optimFile)
print('=> Loading optim from ' .. optimPath)
local optimState = torch.load(optimPath)
local states = torch.load(paths.concat(opt.save, 'states.t7'))
print(states)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestAcc = -math.huge

--[[local states = {}
states.trainLoss = {}
states.testAcc, states.testLoss = {}, {}]]

for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainLoss = trainer:train(epoch, trainLoader) --trainTop1, trainTop5, 
  
   -- Run model on validation set
   local testAcc, testLoss = trainer:test(epoch, valLoader)

   local bestModel = false
   if testAcc > bestAcc then
      bestModel = true
      bestAcc = testAcc
      print(' * Best model Acc: ',testAcc)
   end

   states.trainLoss[epoch] = trainLoss
   states.testAcc[epoch] = testAcc
   states.testLoss[epoch] = testLoss
   -- plot top1 and loss cures
   gnuplot.pngfigure(paths.concat(opt.save,'Acc.png'))
   gnuplot.plot({'testAcc',torch.Tensor(states.testAcc),'-'})
   gnuplot.xlabel('Epoch') 
   gnuplot.ylabel('Acc')
   gnuplot.plotflush()

   
   gnuplot.pngfigure(paths.concat(opt.save,'Loss.png'))
   gnuplot.plot({'train',torch.Tensor(states.trainLoss),'-'}, {'test',torch.Tensor(states.testLoss),'-'})
   gnuplot.xlabel('Epoch') 
   gnuplot.ylabel('loss')
   gnuplot.plotflush()
  
   if epoch % 1 == 0 or epoch == opt.nEpochs then
      checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
   end
    torch.save(paths.concat(opt.save, 'states.t7'), states)
end

print(string.format(' * Finished bestAcc: %7.3f', bestAcc))
