
# modelName = 'speechBaseline4'
# modelName = 'speechAdamW_CosineAnnealingLR'
# modelName = 'speech_Exp2_Linderman'
# modelName = 'speech_Exp3_Augmentations'
# modelName = 'speech_Exp4_FocalLoss'
# modelName = 'speech_Exp5_LSTM_Architecture'
modelName = 'speech_Exp7_Transformer'

import os
current_dir = os.getcwd() 
data_path = os.path.abspath(os.path.join(current_dir,  '..', 'ptDecoder_ctc.pkl'))
output_path = os.path.abspath(os.path.join(current_dir, '..', 'logs', 'speech_logs', modelName))


args = {}
args['outputDir'] = output_path
args['datasetPath'] = data_path
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128
# args['batchSize'] = 8
# args['lrStart'] = 0.05
# args['lrEnd'] = 0.02
args['lrStart'] = 0.001
args['lrEnd'] = 0.0005
args['nUnits'] = 256
args['nBatch'] = 10000 #3000
# args['nBatch'] = 10
# args['nLayers'] = 5

# update layer number for transformer #
args['nLayers'] = 4
##################################

args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
# args['dropout'] = 0.2

# update dropout for transformer #
args['dropout'] = 0.4
##################################

args['whiteNoiseSD'] = 1
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)