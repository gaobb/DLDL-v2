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


local function getImageInfo(list, img_dir)
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

   local trainList = paths.concat(opt.list, 'train15_gt.csv')
   local valList = paths.concat(opt.list, 'valid15_gt.csv')

   local trainDir = 'Align-ChaLearn15/Align.5/Train'
   local valDir =  'Align-ChaLearn15/Align.5/Valid'

   --assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
   --assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)

   print("=> Generating list of images")
   --local classList, classToIdx = findClasses(trainDir)
   print(" | finding all validation images")
   local valImagePath, valImageClass, valImageSigma = getImageInfo(valList, valDir)
   local num_val, len_val = valImagePath:size(1),  valImagePath:size(2)
   local valImageId = torch.range(1, num_val):int()

   print(" | finding all training images")
   local trainImagePath, trainImageClass, trainImageSigma = getImageInfo(trainList, trainDir)
   local num_train, len_train = trainImagePath:size(1), trainImagePath:size(2)	
   local trainImageId = torch.range(1, num_train):int()

   
   local info = {
      basedir = opt.data,
      train = {
         imagePath = trainImagePath,
         imageClass = trainImageClass,
         imageSigma = trainImageSigma,
         imageId = trainImageId,
      },
      val = {
         imagePath = valImagePath,
         imageClass = valImageClass,
         imageSigma = valImageSigma,
         imageId = valImageId,
      },
   }
   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
   
end

return M
