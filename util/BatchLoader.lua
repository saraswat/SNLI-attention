local BatchLoader = {}
local stringx = require('pl.stringx')
BatchLoader.__index = BatchLoader

function BatchLoader.labelToNumber(label)
   local result
   if label == '-' then
      result = 3
      print ('labelToNum sees label -', label)
      -- returning 3 will cause program to abort (it is expecting values 0..2)
      return result
   end
   if label == 'neutral' then
      result = 0
      return result
   end
   if label == 'contradiction' then
      result = 1
      return result
   end
   if label == 'entailment' then
      result = 2
      return result
   end
   print ('labelToNum sees unknown label', label)
   -- returning -1 will cause program to abort.
   return -1
end

-- checkpoint stores word2vec and word2idx. (It also stores
-- idx2word, this is needed for routines that display activation maps
-- for illustrative purposes.)

function BatchLoader.recreate(checkpoint, mode) 
   local o=checkpoint.opt
   return BatchLoader.create(o.data_dir, o.max_length, o.batch_size, checkpoint.vocab, '', mode)
end

function BatchLoader.createScorer(checkpoint, score_files, mode) 
    local o=checkpoint.opt
    local input_files = {}
    x = stringx.split(score_files, ' ')
    for	_, file in pairs(x) do
       table.insert(input_files, file)
    end       
    local self = {}
    setmetatable(self, BatchLoader)
    self.input_files = input_files
    local s1, s2, label, idx2word, word2idx  = BatchLoader.text_to_tensor(input_files, o.max_length, checkpoint.vocab, mode)

    -- Once tensors are created, you no longer need idx2word and word2idx
    -- so we don't need the following assignments
    self.idx2word = idx2word
    self.word2idx = word2idx
    self.vocab_size = #self.idx2word 
--    self.word2vec = load_wordvecs(input_w2v, word2idx)

    return BatchLoader.make_all_batches(self, o.batch_size, s1, s2, label, #input_files)
end

function BatchLoader.create(data_dir, max_sentence_l , batch_size, vocab, vocab_files, mode)
    local train_file = path.join(data_dir, 'train.txt')
    local valid_file = path.join(data_dir, 'dev.txt')
    local test_file = path.join(data_dir, 'test.txt')
    local input_files = {train_file, valid_file, test_file}
    local input_w2v = path.join(data_dir, 'word2vec.txt')
    local self = {}
    setmetatable(self, BatchLoader)
    self.input_files=input_files
    local s1, s2, label, idx2word, word2idx = 
       BatchLoader.text_to_tensor(input_files, max_sentence_l, vocab, mode)
    self.max_sentence_l = max_sentence_l -- vj, I don't think this is needed.
    if vocab_files ~= '' then
       idx2word, word2idx = ingest_vocab_words(vocab_files, idx2word, word2idx, mode)
    end

    -- record it, because we will need to checkpoint this
    self.idx2word, self.word2idx = idx2word, word2idx 
    self.vocab_size = #self.idx2word 
    self.word2vec = load_wordvecs(input_w2v, word2idx)

    return BatchLoader.make_all_batches(self, batch_size, s1, s2, label, 
                                        #input_files)
end

function BatchLoader.make_all_batches(self, batch_size, s1, s2, label, 
                                      num_input_files)
    -- construct a tensor with all the data

    self.split_sizes = {}
    self.all_batches = {}
 
    print(string.format('Word vocab size: %d', #self.idx2word))
    -- cut off the end for train/valid sets so that it divides evenly
    for split=1,num_input_files do
       local s1data = s1[split]
       local s2data = s2[split]
       local label_data = label[split]
       local len = s1data:size(1)
       if len % (batch_size) ~= 0 then
          s1data = s1data:sub(1, batch_size * math.floor(len / batch_size))
          s2data = s2data:sub(1, batch_size * math.floor(len / batch_size))
          --let's just make the batch_size a multiplier of the test data size
          label_data = label_data:sub(1, batch_size * math.floor(len / batch_size)) 
       end

       s1_batches = s1data:split(batch_size,1)
       s2_batches = s2data:split(batch_size,1)
       label_batches = label_data:split(batch_size,1)
       nbatches = #s1_batches
       self.split_sizes[split] = nbatches
       self.all_batches[split] = {s1_batches, s2_batches, label_batches}
    end
 
    self.batch_idx = {}
    print('data load done. Number of batches in train:')
    for i=1,num_input_files do
       table.insert(self.batch_idx, 0)
       print(string.format('File %s %d', self.input_files[i], self.split_sizes[i]))
    end
    collectgarbage()
    return self
end

-- vj: produces
-- output_tensors1[split][s][k] is the index of the k'th word in the s'th
-- sentence of the split file (1-train, 2-validation, 3-test) for T
-- output_tensors2, same, for H
-- labels[split][s] is the label for the data point
-- idx2word is the mapping from index to word
-- word2idx is the mapping from words to index
-- once tensors are produced from input text files, word2idx (and idx2word) 
-- are not needed the model represents a sentence as a sequence of word indices, 
-- and has the mapping rom word index to the corresponding embedding 
-- vector(word_vector)

function BatchLoader.text_to_tensor(input_files, max_sentence_l, vocab, mode)
    print('Processing text into tensors...')

    -- vocab is nil when running cold (not from a checkpoint)
    -- then this routine has to build up id2word and word2idx
    local scoring = vocab ~=nil
    local f
    local vocab_count = {} -- vocab count 
    local idx2word, word2idx
    if mode~='train' then
       idx2word = vocab[1]
       word2idx = vocab[2]
    else 
       idx2word = {'ZERO', 'START'} 
       word2idx = {}; word2idx['ZERO'] = 1; word2idx['START'] = 2
    end
    -- vj we need to add an 'UNK' word for oov words. 

    local split_counts = {}
    local output_tensors1 = {}  --for sentence1
    local output_tensors2 = {}  -- for sentence2
    local labels = {}
    -- first go through train/valid/test to get max sentence length
    -- also counts the number of sentences
    for	split = 1,#input_files do -- split = 1 (train), 2 (val), or 3 (test)
       print('vj: opening file',  input_files[split])
       f = io.open(input_files[split], 'r')       
       local scounts = 0
       for line in f:lines() do
          line1 = line:gsub("^%%.*", "comment")
          if line1=="comment" or line == "" then
          else 
             scounts = scounts + 1
          end
       end
       f:close()
       split_counts[split] = scounts  --the number of sentences in each split
    end

    if #input_files ==3 then
       print(string.format('(T,H) pair count: train %d, val %d, test %d', 
                           split_counts[1], split_counts[2], split_counts[3]))
    else
       print(string.format('(T,H) pair count: test %d', split_counts[1]))
    end
    
    for	split = 1,#input_files do 
       -- split = 1 (train), 2 (val), or 3 (test)     
       -- Preallocate the tensors we will need.
       -- Watch out the second one needs a lot of RAM.
       -- vj: why is the value in the tensor a long, not an int?

       output_tensors1[split] = torch.ones(split_counts[split], max_sentence_l):long() 
       output_tensors2[split] = torch.ones(split_counts[split], max_sentence_l):long() 
       labels[split] = torch.zeros(split_counts[split]):long() 
       -- process each file in split
       f = io.open(input_files[split], 'r')
       local sentence_num = 0
       for line in f:lines() do
          line1 = line:gsub("^%%.*", "comment")
          if line1=="comment" or line == "" then
             --ignore comment and blank lines
          else 
             sentence_num = sentence_num + 1
             local datum = stringx.split(line, '\t')
             local _, s1, s2, label = datum[1], datum[2], datum[3], datum[4]
             if label ~='-' then  -- skip entries with -
                labels[split][sentence_num] = BatchLoader.labelToNumber(label) + 1
                -- append tokens in the sentence1
                output_tensors1[split][sentence_num][1] = 2 -- vj: 'START'
                local word_num = 1
                for rword in s1:gmatch'([^%s]+)' do
                   word_num = word_num + 1
                   if word2idx[rword]==nil then
                      if mode~='train' then
                         print('oov word ', rword, 'replacing with zero for now')
                         rword = 'ZERO'
                      else 
                         idx2word[#idx2word + 1] = rword 
                         word2idx[rword] = #idx2word
                      end
                   end
                   output_tensors1[split][sentence_num][word_num] = word2idx[rword]
                   if word_num == max_sentence_l then break end
                end
                -- append tokens in the sentence2
                output_tensors2[split][sentence_num][1] = 2
                word_num = 1
                for rword in s2:gmatch'([^%s]+)' do
                   word_num = word_num + 1
                   if word2idx[rword]==nil then
                      if mode~='train' then
                         print('oov word ', rword, 'replacing with zero for now')
                         rword = 'ZERO'
                      else 
                         idx2word[#idx2word + 1] = rword 
                         word2idx[rword] = #idx2word
                      end
                   end
                   output_tensors2[split][sentence_num][word_num] = word2idx[rword]
                   if word_num == max_sentence_l then break end
                end
             end
          end
       end
       f:close()
    end
    return output_tensors1, output_tensors2, labels, idx2word, word2idx
end

-- read the named files (csi format), parse for words in T and H
-- add them to word2idx (and idx2word). do not build tensors.
-- this is called when we wish to bulk up word2idx with words that might be
-- encountered during scoring.

function ingest_vocab_words(vocab_files, idx2word, word2idx, mode) 
   assert(mode=='train','vocab files can only be ingested during training from scratch')
   print('loading vocab files...')
   x = stringx.split(vocab_files, ' ')
   for	_, file in pairs(x) do
      print('ingesting from ', file)
       f = io.open(file, 'r')
       for line in f:lines() do
          line1 = line:gsub("^%%.*", "comment")
          if line1=="comment" or line == "" then
             --ignore comment and blank lines
          else 
             local datum = stringx.split(line, '\t')
             local s1, s2 =  datum[2], datum[3] 
             if label ~='-' then  -- skip entries with -
                for rword in s1:gmatch'([^%s]+)' do
                   if word2idx[rword]==nil then
                         idx2word[#idx2word + 1] = rword 
                         word2idx[rword] = #idx2word
                   end
                end
                for rword in s2:gmatch'([^%s]+)' do
                   if word2idx[rword]==nil then
                         idx2word[#idx2word + 1] = rword 
                         word2idx[rword] = #idx2word
                   end
                end
             end
          end
       end
       f:close()
   end
   print('...done')
   return idx2word, word2idx
end

function load_wordvecs(input_w2v, word2idx) 
   -- vj: load up w2v[i] with the 300-long word-vector for the word whose index is i
   print('loading word vecs...')
   local w2v = {}
   local w2v_file = io.open(input_w2v, 'r')
   for line in w2v_file:lines() do
      tokens = stringx.split(line, ' ')
      word = tokens[1]
      if word2idx[word] ~= nil then
         w2v[word2idx[word]] = torch.zeros(300) 
         for tid=2,301 do
            w2v[word2idx[word]][tid-1] = tonumber(tokens[tid])
         end
      end
   end
   w2v_file:close()
   print('...loaded')
   return w2v
end

function BatchLoader:reset_batch_pointer(split_idx, batch_idx)
   batch_idx = batch_idx or 0
   self.batch_idx[split_idx] = batch_idx
end

function BatchLoader:next_batch(split_idx)
    -- split_idx is integer: 1 = train, 2 = val, 3 = test
   self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
   if self.batch_idx[split_idx] > self.split_sizes[split_idx] then
      self.batch_idx[split_idx] = 1 -- cycle around to beginning
   end
   -- pull out the correct next batch
   local idx = self.batch_idx[split_idx]
   return self.all_batches[split_idx][1][idx], self.all_batches[split_idx][2][idx], self.all_batches[split_idx][3][idx]
end


return BatchLoader

