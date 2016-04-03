require 'torch'
require 'nn'
require 'nngraph'
require 'rnn'
model_utils = require 'util.model_utils'
BatchLoader = require 'util.BatchLoader'
-- require 'util.MaskedLoss'
require 'util.misc'
require 'util.CAveTable'
require 'optim'
require 'util.ReplicateAdd'
require 'util.LookupTableEmbedding_train'
classifier_simple = require 'model.classifier_simple'
encoder = require 'model.encoder_lstmn_w2v'
decoder = require 'model.decoder_deep_w2v'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-data_dir', 'data', 'path of the dataset')
cmd:option('-batch_size', '16', 'size of mini-batch')
cmd:option('-max_epochs', 4, 'number of full passes through the training data')
cmd:option('-rnn_size', 400, 'dimensionality of sentence embeddings')
cmd:option('-word_vec_size', 300, 'dimensionality of word embeddings')
cmd:option('-dropout',0.4,'dropout. 0 = no dropout')
cmd:option('-seed',3435,'torch manual random number generator seed')
cmd:option('-max_length', 20, 'max length allowed for T or H, words dropped after that')
cmd:option('-print_every',1000,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every', 12500, 'save epoch')
cmd:option('-checkpoint_dir', 'cv4', 'output directory where checkpoints get written')
cmd:option('-savefile','model','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-checkpoint', 'checkpoint.t7', 'start from a checkpoint if a valid checkpoint.t7 file is given')
cmd:option('-score_files', '', 'file of observations to run scorer on')
cmd:option('-learningRate', 0.001, 'learning rate')
cmd:option('-beta1', 0.9, 'momentum parameter 1')
cmd:option('-beta2', 0.999, 'momentum parameter 2')
cmd:option('-decayRate',0.75,'decay rate for sgd')
cmd:option('-decay_when',0.1,'decay if validation does not improve by more than this much')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-max_grad_norm',5,'normalize gradients at')
cmd:option('-vocab_files','','files to build up vocabulary')
cmd:option('-continue', 0,'continue previous run from checkpoint')
-- GPU/CPU
cmd:option('-gpuid', -1,'which gpu to use. -1 = use CPU')
cmd:option('-cudnn', 0,'use cudnn (1=yes). this should greatly speed up convolutions')
cmd:option('-time', 0, 'print batch times')
cmd:text()

-- checkpoint, score_files: mode = score

-- checkpoint, no scores_file, no -continue: mode=start  (start training from ckp)
-- the train, dev, test files may have nothing to do with the files used to 
-- generate the checkpoint you are starting from. 
-- TODO: permit vocab files to be used to augment the vocabulary as well.

-- checkpoint, no score_file, -continue: mode=continue (previous training run)
-- in this case, supply the same train, dev, test and vocab files as in the
-- previous run

-- no checkpoint, score file: error

-- no checkpoint, no score file: mode=train (training from scratch)

-- parse input params
opt = cmd:parse(arg)
-- load necessary packages depending on config options
if opt.gpuid >= 0 then
   print('using CUDA on GPU ' .. opt.gpuid .. '...')
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.gpuid + 1)
end
if opt.cudnn == 1 then
  assert(opt.gpuid >= 0, 'GPU must be used if using cudnn')
  print('using cudnn...')
  require 'cudnn'
end


opt.mode = 'train'
if opt.score_files ~='' then
   assert(opt.checkpoint ~='checkpoint.t7', 'must specify -checkpoint when scoring')
end
if opt.checkpoint ~='checkpoint.t7' then
   if opt.score_files ~='' then opt.mode='score'
   elseif opt.continue==1 then opt.mode='continue'
   else opt.mode='start'
   end

   print('for mode '.. opt.mode .. ' restoring checkpoint from ' .. opt.checkpoint)
   checkpoint = torch.load(opt.checkpoint)
   print('restored.')

   this_cmdline_opt = opt
   opt = checkpoint.opt -- cmdline is available through this_cmdline_opt.

   -- overwrite some restored option values with info from cmdline.
   opt.mode = this_cmdline_opt.mode
   -- cannot change batch size, must be same as in ckpt
   opt.max_length = this_cmdline_opt.max_length
   opt.data_dir = this_cmdline_opt.data_dir
   opt.checkpoint_dir = this_cmdline_opt.checkpoint_dir
end

torch.manualSeed(opt.seed)

-- create data loader
if opt.mode == 'score' then -- scoring
      loader = BatchLoader.createScorer(checkpoint, this_cmdline_opt.score_files, opt.mode)
elseif opt.mode == 'start' or opt.mode == 'continue' then
   loader = BatchLoader.recreate(checkpoint, opt.mode)
else --train
   loader = BatchLoader.create(opt.data_dir, opt.max_length, opt.batch_size, 
                               nil, opt.vocab_files, opt.mode)
   opt.seq_length = loader.max_sentence_l 
   opt.vocab_size = #loader.idx2word
   opt.classes = 3
   opt.word2vec = loader.word2vec
end

-- model
if opt.mode ~='train' then
   protos = checkpoint.protos -- recover model from checkpoint
else 
   protos = {}
   protos.enc = encoder.lstmn(opt.vocab_size, opt.rnn_size, opt.dropout, 
                              opt.word_vec_size, opt.batch_size, opt.word2vec) 
   protos.dec = decoder.lstmn(opt.vocab_size, opt.rnn_size, opt.dropout, 
                              opt.word_vec_size, opt.batch_size, opt.word2vec) 
   protos.criterion = nn.ClassNLLCriterion()
   protos.classifier = classifier_simple.classifier(opt.rnn_size, opt.dropout, 
                                                    opt.classes)
end

-- ship to gpu
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-- params and grads
params, grad_params = model_utils.combine_all_parameters(protos.enc, protos.dec, 
                                                         protos.classifier)
print('number of parameters in the model: ' .. params:nElement())

function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'enc_lookup' then
      enc_lookup = layer
    elseif layer.name == 'dec_lookup' then
      dec_lookup = layer
    end
  end
end

protos.enc:apply(get_layer)
protos.dec:apply(get_layer)
--dec_lookup:share(enc_lookup, 'weight')

-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
  if name == 'enc' or name == 'dec' then
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.max_length, 
                                                   not proto.parameters)
  end
end
-- encoder/decoder initial states, decoder initial alignment vector
local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
if opt.gpuid >=0 then h_init = h_init:cuda() end
assert(h_init ~=nil, 'LSTMN: 171 h_init cannot be nil ')

-- main
if opt.mode=='score' then  -- scoring
   print('scoring...')
   -- note: scorer does not create checkpoints.
   for i=1,#loader.split_sizes do
      test_loss = model_utils.eval_split(opt, loader, protos, clones, h_init, i)
      print (string.format("File %s test_loss = %6.4f", loader.input_files[i], 
                           test_loss))
   end
else 
   max_epochs=opt.max_epochs
   if opt.mode == 'train' then print('training...')
   elseif opt.mode == 'continue' then print('continuing previous training run...')
   else print('starting training from ckp...'); max_epochs=this_cmdline_opt.max_epochs
   end
   model_utils.train(checkpoint, opt, loader, protos, clones, h_init, 
                     params, grad_params, max_epochs) 
   test_accuracy = model_utils.eval_split(opt, loader, protos, clones, h_init, 3)
   print (string.format("test_accuracy = %6.4f", test_accuracy))
end

