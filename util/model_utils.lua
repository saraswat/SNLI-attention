
-- adapted from https://github.com/wojciechz/learning_to_execute
-- utilities for combining/flattening parameters in a model
-- the code in this script is more general than it needs to be, which is 
-- why it is kind of a large

require 'torch'
local model_utils = {}
function model_utils.combine_all_parameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- get parameters
    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
        local tn = torch.typename(layer)
	local net_params, net_grads = networks[i]:parameters()

	if net_params then
	    for _, p in pairs(net_params) do
		parameters[#parameters + 1] = p
	    end
	    for _, g in pairs(net_grads) do
		gradParameters[#gradParameters + 1] = g
	    end
	end
    end

    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end

function model_utils.clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

function model_utils.eval_split(opt, loader, protos, clones, h_init, split_idx)
  print('evaluating loss over split index ', split_idx)
  assert(h_init ~=nil, 'model_utils: 161 h_init cannot be nil ')
  local n = loader.split_sizes[split_idx]
  loader:reset_batch_pointer(split_idx)
  local correct_count = 0
  for i = 1,n do
    -- load data
    local x, y, label = loader:next_batch(split_idx)
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
      local lst = clones.enc[t]:forward({x[{{},t}], narrow_list(rnn_c_enc, 1, t), 
                                         narrow_list(rnn_h_enc, 1, t)})
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
      local lst = clones.dec[t]:forward({y[{{},t}], rnn_a[t-1], rnn_alpha[t-1], 
                                         narrow_list(rnn_c_dec, 1, t), 
                                         narrow_list(rnn_h_dec, 1, t), 
                                         rnn_c_enc, rnn_h_enc})
      table.insert(rnn_a, lst[1])
      table.insert(rnn_alpha, lst[2])
      table.insert(rnn_c_dec, lst[3])
      table.insert(rnn_h_dec, lst[4])
    end
    -- 3) classification
    protos.classifier:evaluate()
    local prediction = protos.classifier:forward({rnn_h_enc, rnn_h_dec})
    local max,indice = prediction:max(2)   -- indice is a 2d tensor here, we need to flatten it...
    for i=1, opt.batch_size do
       if indice[i][1] == label[i] then correct_count = correct_count + 1 end
    end
  end
  return correct_count*1.0/(n*opt.batch_size)
end

function model_utils.train(checkpoint, opt, loader, protos, clones, h_init, 
                           params, grad_params,max_epochs) 
   assert(h_init ~=nil, 'model_utils 219 h_init must not be null')
   feval = fevalMaker(opt, loader, protos, clones, h_init, params, grad_params)
   assert(h_init ~=nil, 'model_utils 222 h_init must not be null')
   if opt.mode =='continue' then 
      train_losses = checkpoint.train_losses
      val_losses = checkpoint.val_losses
      start_iterations = checkpoint.train_next_iter
      iterations = max_epochs * loader.split_sizes[1]
   else 
      train_losses = {}
      val_losses = {}
      start_iterations=1
      iterations = max_epochs*loader.split_sizes[1]
   end

   local optim_state = {learningRate = opt.learningRate, beta1 = opt.beta1, beta2 = opt.beta2}
   print('start_iterations ', start_iterations, ' iterations ', iterations)
   for i = start_iterations, iterations do
      -- train 
      local epoch = i / loader.split_sizes[1]
      local timer = torch.Timer()
      local time = timer:time().real
      local _, loss = optim.adam(feval, params, optim_state)
      train_losses[i] = loss[1]
      if i % opt.print_every == 0 then
         print(string.format("%d/%d (epoch %.2f), train_loss = %6.4f", i, iterations, epoch, train_losses[i]))
      end

      -- validate and save checkpoints
      if epoch == max_epochs or i % opt.save_every == 0 then
         assert(h_init ~=nil, 'model_utils 250 h_init must not be null')
         print ('evaluate on validation set')
         local val_loss = model_utils.eval_split(opt, loader, protos, clones, h_init,  2) -- 2 = validation
         print (val_loss)
         if epoch>1.5 then
            assert(h_init ~=nil, 'model_utils 255 h_init must not be null')
            local test_loss = model_utils.eval_split(opt, loader, protos, clones, h_init, 3) -- 3 = test
            print (test_loss)
         end
         val_losses[#val_losses+1] = val_loss
         local savefile = string.format('%s/model_%s_epoch%.2f_%.2f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
         local checkpoint = {}
         checkpoint.protos = protos
         checkpoint.opt = opt
         checkpoint.train_next_iter = i+1
         checkpoint.train_losses = train_losses
         checkpoint.val_losses = val_losses
         checkpoint.vocab = {loader.idx2word, loader.word2idx}
         print('saving checkpoint to ' .. savefile)
         torch.save(savefile, checkpoint)
      end

      -- decay learning rate
      if i % loader.split_sizes[1] == 0 and #val_losses > 2 then
         if val_losses[#val_losses-1] - val_losses[#val_losses] < opt.decay_when then
            opt.learningRate = opt.learningRate * opt.decayRate
         end
      end

      -- index 1 is zero
      enc_lookup.weight[1]:zero()
      enc_lookup.gradWeight[1]:zero()
      dec_lookup.weight[1]:zero()
      dec_lookup.gradWeight[1]:zero()

      -- misc
      if i%5==0 then collectgarbage() end
      if opt.time ~= 0 then
         print("Batch Time:", timer:time().real - time)
      end
   end
end

function fevalMaker(opt, loader, protos, clones, h_init, params, grad_params)
   return function(x)
      if x ~= params then
         params:copy(x)
      end
      grad_params:zero()
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
      assert(h_init ~=nil, 'model_utils 312 h_init must not be null')
      table.insert(rnn_c_enc, h_init:clone())
      table.insert(rnn_h_enc, h_init:clone())
      for t=1,opt.max_length do
         clones.enc[t]:training()
         local lst = clones.enc[t]:forward({x[{{},t}], narrow_list(rnn_c_enc, 1, t), narrow_list(rnn_h_enc, 1, t)})
         table.insert(rnn_c_enc, lst[1])
         table.insert(rnn_h_enc, lst[2])
      end
      -- 2) decoder
      local rnn_c_dec = {}
      local rnn_h_dec = {}
      assert(h_init ~=nil, 'model_utils 324 h_init must not be null')
      local rnn_a = {[0] = h_init:clone()}
      local rnn_alpha = {[0] = h_init:clone()}
      assert(h_init ~=nil, 'model_utils 327 h_init must not be null')
      table.insert(rnn_c_dec, rnn_c_enc[opt.max_length+1]:clone())
      table.insert(rnn_h_dec, rnn_h_enc[opt.max_length+1]:clone())
      for t=1,opt.max_length do
         clones.dec[t]:training()
         local lst = clones.dec[t]:forward({y[{{},t}], rnn_a[t-1], rnn_alpha[t-1], narrow_list(rnn_c_dec, 1, t), narrow_list(rnn_h_dec, 1, t), rnn_c_enc, rnn_h_enc})
         table.insert(rnn_a, lst[1])
         table.insert(rnn_alpha, lst[2])
         table.insert(rnn_c_dec, lst[3])
         table.insert(rnn_h_dec, lst[4])
      end
      -- 3) classification
      protos.classifier:training()
      local prediction = protos.classifier:forward({rnn_h_enc, rnn_h_dec})
      local result = protos.criterion:forward(prediction, label)

      -- Backward pass
      -- 1) classification
      local dresult = protos.criterion:backward(prediction, label)
      local dprediction = protos.classifier:backward({rnn_h_enc, rnn_h_dec}, dresult)
      local drnn_alpha = clone_list(rnn_a, true) --true zeros
      local drnn_a = clone_list(rnn_a, true)
      local drnn_c_enc = clone_list(rnn_c_enc, true)
      local drnn_h_enc = clone_list(rnn_h_enc, true)
      local drnn_c_dec = clone_list(rnn_c_dec, true)
      local drnn_h_dec = clone_list(rnn_h_dec, true)

      for t=1,opt.max_length+1 do
         drnn_h_enc[t]:add(dprediction[1][t])
         drnn_h_dec[t]:add(dprediction[2][t])
      end

      -- 2) decoder
      for t=opt.max_length,1,-1 do
         local dlst = clones.dec[t]:backward({y[{{},t}], rnn_a[t-1], rnn_alpha[t-1], narrow_list(rnn_c_dec, 1, t), narrow_list(rnn_h_dec, 1, t), rnn_c_enc, rnn_h_enc}, {drnn_a[t], drnn_alpha[t], drnn_c_dec[t+1], drnn_h_dec[t+1]})
         drnn_a[t-1]:add(dlst[2])
         drnn_alpha[t-1]:add(dlst[3])
         for k=1, t do
            drnn_c_dec[k]:add(dlst[4][k])    
            drnn_h_dec[k]:add(dlst[5][k])    
         end
         for k=1, opt.max_length+1 do
            drnn_c_enc[k]:add(dlst[6][k])
            drnn_h_enc[k]:add(dlst[7][k])
         end
      end

      -- 3) encoder
      drnn_c_enc[opt.max_length+1]:add(drnn_c_dec[1])
      drnn_h_enc[opt.max_length+1]:add(drnn_h_dec[1])
      for t=opt.max_length,1,-1 do
         dlst = clones.enc[t]:backward({x[{{},t}], narrow_list(rnn_c_enc, 1, t), narrow_list(rnn_h_enc, 1, t)}, {drnn_c_enc[t+1], drnn_h_enc[t+1]})
         for k=1, t do
            drnn_c_enc[k]:add(dlst[2][k])    
            drnn_h_enc[k]:add(dlst[3][k])    
         end
      end

      local grad_norm, shrink_factor
      grad_norm = torch.sqrt(grad_params:norm()^2)
      if grad_norm > opt.max_grad_norm then
         shrink_factor = opt.max_grad_norm / grad_norm
         grad_params:mul(shrink_factor)
      end
      assert(h_init ~=nil, 'model_utils 391 h_init must not be null')
      return result, grad_params 
   end  -- end of feval 
end

return model_utils
