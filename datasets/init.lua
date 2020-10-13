--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet and CIFAR-10 datasets
--
--   matio = require 'matio'

local M = {}

local function isvalid(opt, cachePath)
   local imageInfo = torch.load(cachePath)
   if imageInfo.basedir and imageInfo.basedir ~= opt.data then
      return false
   end
   return true
end

function M.create(opt, split)
   local cachePath = paths.concat(opt.gen, opt.dataset .. '.t7')
   print("path: " .. cachePath)
   if not paths.filep(cachePath) then --or not isvalid(opt, cachePath) then
      paths.mkdir('gen')
      local script = paths.dofile(opt.dataset .. '-gen.lua')
      script.exec(opt, cachePath)
   end
   local imageInfo = torch.load(cachePath)

   --local vgg_feats = matio.load(paths.concat(opt.gen, 'chalearn15_vggface_feats.mat'))
   --imageInfo.train.cnnfeats = vgg_feats.train.vggfacefeats:type('torch.FloatTensor')
   --imageInfo.val.cnnfeats = vgg_feats.val.vggfacefeats:type('torch.FloatTensor')
   local Dataset = require('datasets/' .. opt.dataset)
   return Dataset(imageInfo, opt, split)
end

return M
