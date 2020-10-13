--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet dataset loader
--
--local matio = require 'matio'
local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local ImagenetDataset = torch.class('resnet.ImagenetDataset', M)

function ImagenetDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = imageInfo.basedir
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function ImagenetDataset:get(i)
   local path = ffi.string(self.imageInfo.imagePath[i]:data())
   local image = self:_loadImage(paths.concat(self.dir, path))
   local class = self.imageInfo.imageClass[i]
   local sigma = self.imageInfo.imageSigma[i]
   local mask = self.imageInfo.imageMask[i]
   local id = self.imageInfo.imageId[i]

   return {
      input = image,
      target = class,
      mask = mask,
      sigma = sigma,
      id = id,
   }
end

function ImagenetDataset:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      print(path)
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, 3, 'float')
   end

   return input
end

function ImagenetDataset:size()
   return self.imageInfo.imageClass:size(1)
   --return self.imageInfo.age:size(1)
end

-- Computed from random subset of Age training images(chalearn16 and google faces)
local meanstd = {
   mean = { 0.5958, 0.4637, 0.4065 },
   std =  { 0.2693, 0.2409, 0.2352 },
}     
function ImagenetDataset:preprocess()
   if self.split == 'train' then
      if self.opt.dataAug == 'true' then 
         return t.Compose{
         t.HorizontalFlip(0.5),
         t.Gray(0.5),
         t.RandomScale(112, 224),
         t.Rotation(30),
         t.Scale(224),
         t.RandomCrop(224),
         t.ColorJitter({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
         }),
         t.ColorNormalize(meanstd),
      }
      else
         return t.Compose{
         t.Scale(224),
         t.CenterCrop(224),
         t.ColorNormalize(meanstd),
      }
      end
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.Scale(224),
         t.CenterCrop(224),
         t.ColorNormalize(meanstd),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.ImagenetDataset
