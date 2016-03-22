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
-- cmd:option('-max_epochs', 4, 'number of full passes through the training data')
cmd:option('-rnn_size', 400, 'dimensionality of sentence embeddings')
cmd:option('-word_vec_size', 300, 'dimensionality of word embeddings')
-- cmd:option('-dropout',0.4,'dropout. 0 = no dropout')
cmd:option('-seed',3435,'torch manual random number generator seed')
cmd:option('-max_length', 20, 'max length allowed for each sentence')
cmd:option('-checkpoint', 'checkpoint.t7', 'start from a checkpoint if a valid checkpoint.t7 file is given')
-- cmd:option('-learningRate', 0.001, 'learning rate')
-- cmd:option('-beta1', 0.9, 'momentum parameter 1')
-- cmd:option('-beta2', 0.999, 'momentum parameter 2')
-- cmd:option('-decayRate',0.75,'decay rate for sgd')
-- cmd:option('-decay_when',0.1,'decay if validation does not improve by more than this much')
-- cmd:option('-param_init', 0.05, 'initialize parameters at')
-- cmd:option('-max_grad_norm',5,'normalize gradients at')
-- GPU/CPU
cmd:option('-gpuid', -1,'which gpu to use. -1 = use CPU')
cmd:option('-cudnn', 0,'use cudnn (1=yes). this should greatly speed up convolutions')
-- cmd:option('-time', 0, 'print batch times')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

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

restarting_from_checkpoint=true
print('restoring checkpoint from ' .. opt.checkpoint)
checkpoint = torch.load(opt.checkpoint)
this_cmdline_opt = opt
opt = checkpoint.opt -- thus all options other than checkpoint are ignored from cmdline
print('restored.')

-- create data loader
loader = BatchLoader.createScorer(opt.data_dir, opt.max_length, opt.batch_size)
opt.seq_length = loader.max_sentence_l 
opt.vocab_size = #loader.idx2word
opt.classes = 3
opt.word2vec = loader.word2vec

-- model
protos = checkpoint.protos

-- ship to gpu
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end
-- params and grads
-- vj: I believe these can be deleted.
params, grad_params = model_utils.combine_all_parameters(protos.enc, protos.dec, protos.classifier)
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
-- vj is this needed for scoring?
protos.enc:apply(get_layer)
protos.dec:apply(get_layer)
--dec_lookup:share(enc_lookup, 'weight')

-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
  if name == 'enc' or name == 'dec' then
    clones[name] = model_utils.clone_many_times(proto, opt.max_length, not proto.parameters)
  end
end
-- encoder/decoder initial states, decoder initial alignment vector
local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
if opt.gpuid >=0 then h_init = h_init:cuda() end


--evaluation 
function eval_split()
   print('vj: evaluating')
  local n = loader.split_sizes[1]
  loader:reset_batch_pointer(1)
  local correct_count = 0
  for i = 1,n do
    -- load data
    local x, y, label = loader:next_batch(1)
    if opt.gpuid >= 0 then
      x = x:float():cuda()
      y = y:float():cuda()
      label = label:float():cuda()
    end

    -- Forward pass
    -- 1) encoder
    local rnn_c_enc = {}
    local rnn_h_enc = {}
    table.insert(rnn_c_enc, h_init:clone())
    table.insert(rnn_h_enc, h_init:clone())
    for t=1,opt.max_length do
      clones.enc[t]:evaluate()
      local lst = clones.enc[t]:forward({x[{{},t}], narrow_list(rnn_c_enc, 1, t), narrow_list(rnn_h_enc, 1, t)})
      table.insert(rnn_c_enc, lst[1])
      table.insert(rnn_h_enc, lst[2])
    end
    -- 2) decoder
    local rnn_c_dec = {}
    local rnn_h_dec = {}
    local rnn_a = {[0] = h_init:clone()}
    local rnn_alpha = {[0] = h_init:clone()}
    table.insert(rnn_c_dec, rnn_c_enc[opt.max_length+1]:clone())
    table.insert(rnn_h_dec, rnn_h_enc[opt.max_length+1]:clone())
    for t=1,opt.max_length do
      clones.dec[t]:evaluate()
      local lst = clones.dec[t]:forward({y[{{},t}], rnn_a[t-1], rnn_alpha[t-1], narrow_list(rnn_c_dec, 1, t), narrow_list(rnn_h_dec, 1, t), rnn_c_enc, rnn_h_enc})
      table.insert(rnn_a, lst[1])
      table.insert(rnn_alpha, lst[2])
      table.insert(rnn_c_dec, lst[3])
      table.insert(rnn_h_dec, lst[4])
    end
    -- 3) classification
    protos.classifier:evaluate()
    local prediction = protos.classifier:forward({rnn_h_enc, rnn_h_dec})
    if ( 0 == (i%50)) then
       print('prediction ', prediction)
       print('i=', i, ' prediction=', indice[1][1], ' label=', label[1])
    end
    local max,indice = prediction:max(2)   -- indice is a 2d tensor here, we need to flatten it...
    if indice[1][1] == label[1] then correct_count = correct_count + 1 end
  end
  return correct_count*1.0/n
end

test_loss = eval_split()
print (string.format("accuracy = %6.4f", test_loss))
