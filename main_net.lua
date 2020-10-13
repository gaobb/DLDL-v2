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
--local matio = require 'matio'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train_net'
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

if opt.dataset == 'scut-fbp' or 
   opt.dataset == 'scut-fbp5500' or 
   opt.dataset == 'scut-fbp5500_1' or 
   opt.dataset == 'scut-fbp5500_2' or 
   opt.dataset == 'scut-fbp5500_3' or 
   opt.dataset == 'scut-fbp5500_4' or 
   opt.dataset == 'scut-fbp5500_5' then
   --opt.labelStep = 0.1
   opt.labelSet = torch.range(1, 5, opt.labelStep)
elseif opt.dataset == 'cfd' then
   --opt.labelStep = 0.1
   opt.labelSet = torch.range(1, 7, opt.labelStep)
elseif opt.dataset == 'hotnot' then
   --opt.labelStep = 0.1
   opt.labelSet = torch.range(-3.6, 3.6, opt.labelStep)
elseif opt.dataset == 'selfie' or opt.dataset == 'allselfie' then
   --opt.labelStep = 0.1
   opt.labelSet = torch.range(1.5, 7, opt.labelStep)
elseif opt.dataset == 'boneage' or opt.dataset == 'alignboneage' then
   --opt.labelStep = 1
   opt.labelSet = torch.range(1, 230, opt.labelStep)
else
   --opt.labelStep = 1
   opt.labelSet = torch.range(0, 100, opt.labelStep)
end


local savePath = paths.concat('Training-Models',opt.dataset, opt.loss..'-'..opt.netType..'-CR'..opt.CR..'-Aug'..opt.dataAug..'-Step'..opt.labelStep)
opt.save = savePath 
print(opt.save)

if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
end
-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

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
local bestExpMae = math.huge
local bestMaxMae = math.huge

local states = {}
states.trainExpMae, states.trainMaxMae, states.trainLoss = {}, {}, {}
states.testExpMae, states.testMaxMae, states.testLoss = {}, {}, {}

for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   print(epoch)
   
   local trainMaxMae, trainExpMae, trainLoss = trainer:train(epoch, trainLoader) --trainTop1, trainTop5,  
  
   -- Run model on validation set
   local testMaxMae, testExpMae, testLoss = trainer:test(epoch, valLoader)

   local bestModel = false
   if testExpMae < bestExpMae then
      bestModel = true
      bestExpMae = testExpMae
      bestMaxMae = testMaxMae
      print(' * Best model, maxMae: ',testMaxMae, 'expMae: ',testExpMae)
   end

   states.trainLoss[epoch] = trainLoss
   states.trainExpMae[epoch] = trainExpMae
   states.trainMaxMae[epoch] = trainMaxMae
   states.testExpMae[epoch] = testExpMae
   states.testMaxMae[epoch] = testMaxMae
   states.testLoss[epoch] = testLoss
   -- plot top1 and loss cures
   gnuplot.pngfigure(paths.concat(opt.save,'MAE_Error.png'))
   gnuplot.plot({'trainExpMae',torch.Tensor(states.trainExpMae),'-'}, {'trainMaxMae', torch.Tensor(states.trainMaxMae),'-'},{'testExpMae',torch.Tensor(states.testExpMae),'-'}, {'testMaxMae', torch.Tensor(states.testMaxMae),'-'})
   gnuplot.xlabel('Epoch') 
   gnuplot.ylabel('MAE')
   gnuplot.plotflush()

   
   gnuplot.pngfigure(paths.concat(opt.save,'loss.png'))
   gnuplot.plot({'train',torch.Tensor(states.trainLoss),'-'}, {'test',torch.Tensor(states.testLoss),'-'})
   gnuplot.xlabel('Epoch') 
   gnuplot.ylabel('loss')
   gnuplot.plotflush()
  
   if epoch % 30 == 0 or epoch == opt.nEpochs then
       checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
       torch.save(paths.concat(opt.save, 'states.t7'), states)
   end
end

print(string.format(' * Finished bestmaxMae: %7.3f bestexpMae: %7.3f', bestMaxMae, bestExpMae))
