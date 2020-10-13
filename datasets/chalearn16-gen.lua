--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This generates a file gen/imagenet.t7 which contains the list of all
--  ImageNet training and validation images and their classes. This script also
--  works for other datasets arragned with the same layout.
--

local sys = require 'sys'
local ffi = require 'ffi'

local M = {}

local function findClasses(dir)
   local dirs = paths.dir(dir)
   table.sort(dirs)

   local classList = {}
   local classToIdx = {}
   for _ ,class in ipairs(dirs) do
      if not classToIdx[class] and class ~= '.' and class ~= '..' and class ~= '.DS_Store' then
         table.insert(classList, class)
         classToIdx[class] = #classList
      end
   end

   -- assert(#classList == 1000, 'expected 1000 ImageNet classes')
   return classList, classToIdx
end

local function findImages(dir, classToIdx)
   local imagePath = torch.CharTensor()
   local imageClass = torch.LongTensor()

   ----------------------------------------------------------------------
   -- Options for the GNU and BSD find command
   local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- Find all the images using the find command
   local f = io.popen('find -L ' .. dir .. findOptions)

   local maxLength = -1
   local imagePaths = {}
   local imageClasses = {}

   -- Generate a list of all the images and their class
   while true do
      local line = f:read('*line')
      if not line then break end

      local className = paths.basename(paths.dirname(line))
      local filename = paths.basename(line)
      local path = className .. '/' .. filename

      local classId = classToIdx[className]
      assert(classId, 'class not found: ' .. className)

      table.insert(imagePaths, path)
      table.insert(imageClasses, classId)

      maxLength = math.max(maxLength, #path + 1)
   end

   f:close()

   -- Convert the generated list to a tensor for faster loading
   local nImages = #imagePaths
   local imagePath = torch.CharTensor(nImages, maxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end

   local imageClass = torch.LongTensor(imageClasses)
   return imagePath, imageClass
end


local function readImageInfo(list, img_dir)
   local imagePath = torch.CharTensor()
   local imageClass = torch.FloatTensor()
   local imageSigma = torch.FloatTensor()

   local maxLength = -1
   local imagePaths = {}
   local imageClasses = {}
   local imageSigmas = {}

   local file = io.open(list, 'r')
   for line in file:lines() do
      local  path0, age, sigma = unpack(line:split(","))
      local  path = img_dir ..'/'.. path0
      table.insert(imagePaths, path)
      table.insert(imageClasses, age)
      table.insert(imageSigmas, sigma)
      maxLength = math.max(maxLength, #path + 1)
   end
   file:close()
    
   -- Convert the generated list to a tensor for faster loading
   local nImages = #imagePaths
   local imagePath = torch.CharTensor(nImages, maxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end

   local imageClass = torch.FloatTensor(imageClasses)
   local imageSigma = torch.FloatTensor(imageSigmas)
   return imagePath, imageClass, imageSigma
end

function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)
   opt.data = "./Data/images"
   opt.list = "./Data/train-val-list"

   local trainList = paths.concat(opt.list, 'train16_gt.csv')
   local testList = paths.concat(opt.list, 'test16_gt.csv')
   local valList = paths.concat(opt.list, 'valid16_gt.csv')

   local trainDir = 'Align-ChaLearn16/Align.5/Train'
   local testDir = 'Align-ChaLearn16/Align.5/Test'
   local valDir =  'Align-ChaLearn16/Align.5/Valid'

   --assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
   --assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)

   print("=> Generating list of images")
   --local classList, classToIdx = findClasses(trainDir)
   print(" | finding all validation images")
   local valImagePath, valImageClass, valImageSigma = readImageInfo(valList, valDir)
   local num_val, len_val = valImagePath:size(1),  valImagePath:size(2)
   local valImageMask = torch.ByteTensor(num_val):fill(1)

   print(" | finding all training images")
   local trainImagePath, trainImageClass, trainImageSigma = readImageInfo(trainList, trainDir)
   local num_train, len_train = trainImagePath:size(1), trainImagePath:size(2)	
   local trainImageMask = torch.ByteTensor(num_train):fill(1)

   print(" | finding all testing images")
   local testImagePath, testImageClass, testImageSigma = readImageInfo(testList, testDir)
   local num_test, len_test = testImagePath:size(1), testImagePath:size(2)	
   local testImageMask = torch.ByteTensor(num_test):fill(1)
   local testImageId = torch.range(1, num_test):int()

   
    -- merge train and val data to form training data
   local num_Train, len_Train = num_train + num_val, math.max(len_train, len_val)
   local TrainImagePath = torch.CharTensor(num_Train, len_Train):zero()
   
   -- train
   local temp = torch.CharTensor(num_train,  len_Train):zero()
   temp:indexCopy(2, torch.range(1, len_train):type('torch.LongTensor'), trainImagePath)
   TrainImagePath:indexCopy(1, torch.range(1, num_train):type('torch.LongTensor'), temp)
   -- val
   temp = torch.CharTensor(num_val,  len_Train):zero()
   temp:indexCopy(2, torch.range(1, len_val):type('torch.LongTensor'), valImagePath)
   TrainImagePath:indexCopy(1, torch.range(num_train + 1, num_train + num_val):type('torch.LongTensor'), temp)

   local TrainImageClass = torch.cat(trainImageClass, valImageClass)
   local TrainImageSigma = torch.cat(trainImageSigma, valImageSigma)
   local TrainImageMask =  torch.cat(trainImageMask,  valImageMask)
   local TrainImageId = torch.range(1, num_Train):int()
   
   local info = {
      basedir = opt.data,
      --classList = classList,
      train = {
         imagePath = TrainImagePath,
         imageClass = TrainImageClass,
         imageSigma = TrainImageSigma,
         imageMask = TrainImageMask,
         imageId = TrainImageId,
      },
      val = {
         imagePath = testImagePath,
         imageClass = testImageClass,
         imageSigma = testImageSigma,
         imageMask = testImageMask,
         imageId = testImageId,
      },
   }
   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
   
end

return M





--[[-- resampling
   local roundclass =  TrainImageClass:clone():round()
   local expNum = 150
   local praNum = 0
   local ids = {}
   for c = 1, 100 do
       print(c)
       local cid = torch.IntTensor(torch.find(roundclass, c))
       if cid:dim() >= 1 then
       praNum = cid:size(1)
       local resid
       if praNum > 0  and praNum < 150 then
          local t = math.floor(expNum/praNum)
          local r = expNum % praNum
          if r > 0 then
             local perm = torch.randperm(praNum)
             local rid = perm:narrow(1, 1, r)
             resid = torch.cat(torch.repeatTensor(cid, 1, t):transpose(1,2), rid:int(), 1)
          else
             resid = torch.repeatTensor(cid, 1, t):transpose(1,2)
          end
      elseif  praNum >= 150 then
             resid = cid:reshape(praNum,1)
      end
      table.insert(ids, resid)
      else 
      end
   end
   local reids = torch.cat(ids, 1):squeeze():long(
   
   
   local info = {
      basedir = opt.data,
      --classList = classList,
      train = {
         imagePath = TrainImagePath:index(1, reids),
         imageClass = TrainImageClass:index(1, reids),
         imageSigma = TrainImageSigma:index(1, reids),
         imageMask = TrainImageMask:index(1, reids),
         imageId = torch.range(1, reids:size(1)):int(),
      },
      val = {
         imagePath = testImagePath,
         imageClass = testImageClass,
         imageSigma = testImageSigma,
         imageMask = testImageMask,
         imageId = testImageId,
      },
   }]]
