-- Training
CUDA_VISIBLE_DEVICES=14,15 th main_net.lua -dataset chalearn15 -nGPU 2 -batchSize 128 -dataAug true  -nEpochs 60 -loss ldkl -LR 0.001 -netType dldlnet-msceleb1m -CR 0.5 -labelStep 1
CUDA_VISIBLE_DEVICES=14,15 th main_mtnet.lua -dataset chalearn15 -nGPU 2 -batchSize 128 -dataAug true -nEpochs 60 -loss ldklexpl1 -LR 0.001 -netType mtdldlnet-msceleb1m  -CR 0.5 -labelStep 1 -lambda 1



CUDA_VISIBLE_DEVICES=12,13 th main_net.lua -dataset scut-fbp -nGPU 2 -batchSize 128 -dataAug true  -nEpochs 60 -loss ldkl  -LR 0.001 -netType dldlnet-msceleb1m -CR 0.5 -labelStep 0.1
CUDA_VISIBLE_DEVICES=14,15 th main_net.lua -dataset scut-fbp -nGPU 2 -batchSize 128 -dataAug true  -nEpochs 60 -loss expl1 -LR 0.001 -netType dldlnet-msceleb1m -CR 0.5 -labelStep 0.1



-- Evaluation
CUDA_VISIBLE_DEVICES=1 th evaluation.lua -dataset chalearn15 -loss ldklexp -netType mtdldlnet-msceleb1m -CR 0.5 -dataAug true  -labelStep 1

CUDA_VISIBLE_DEVICES=1 th evaluation.lua -dataset chalearn15 -loss ldklexpl1 -netType mtdldlnet-msceleb1m -CR 0.5 -dataAug true -labelStep 1 -lambda 1







