local ScorerBatchloader = {}
local stringx = require('pl.stringx')
ScorerBatchloader.__index = ScorerBatchloader

function ScorerBatchloader.labelToNumber(label)
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

function ScorerBatchloader.create(data_dir, max_sentence_l , batch_size)
    local self = {}
    setmetatable(self, ScorerBatchloader)
    local test_file = path.join(data_dir, 'test.txt')
    local input_files = {train_file, valid_file, test_file}
    local input_w2v = path.join(data_dir, 'word2vec.txt')
    

    -- construct a tensor with all the data
    local s1, s2, label, idx2word, word2idx, word2vec = ScorerBatchloader.text_to_tensor(test_file, max_sentence_l, input_w2v)
    self.max_sentence_l = max_sentence_l
    self.idx2word, self.word2idx = idx2word, word2idx
    self.vocab_size = #self.idx2word 
    self.word2vec = word2vec
    self.split_size = 0
    self.all_batches = {}
 
    print(string.format('Word vocab size: %d', #self.idx2word))
    local s1data = s1
    local s2data = s2
    local label_data = label
    local len = s1data:size(1)
    if len % (batch_size) ~= 0 then
       s1data = s1data:sub(1, batch_size * math.floor(len / batch_size))
       s2data = s2data:sub(1, batch_size * math.floor(len / batch_size))
       label_data = label_data:sub(1, batch_size * math.floor(len / batch_size))   --let's just make the batch_size a multiplier of the test data size
    end

    s1_batches = s1data:split(batch_size,1)
    s2_batches = s2data:split(batch_size,1)
    label_batches = label_data:split(batch_size,1)
    nbatches = #s1_batches
    self.split_size = nbatches
    self.all_batches = {s1_batches, s2_batches, label_batches}
 
    self.batch_idx = 0
    print(string.format('data load done. Number of batches in train: %d', self.split_size))
    collectgarbage()
    return self
end

function ScorerBatchloader:reset_batch_pointer(batch_idx)
    batch_idx = batch_idx or 0
    self.batch_idx = batch_idx
end

function ScorerBatchloader:next_batch()
    self.batch_idx = self.batch_idx + 1
    if self.batch_idx > self.split_size then
        self.batch_idx = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local idx = self.batch_idx
    return self.all_batches[1][idx], self.all_batches[2][idx], self.all_batches[3][idx]
end

-- vj: produces
-- output_tensors1[split][s][k] is the index of the k'th word in the s'th
-- sentence of the split file (1-train, 2-validation, 3-test) for T
-- output_tensors2, same, for H
-- labels[split][s] is the label for the data point
-- idx2word is the mapping from index to word
-- word2idx is the mapping from words to index
-- w2v is the mapping from index to the vector (embedding) for the word

function ScorerBatchloader.text_to_tensor(test_file, max_sentence_l, input_w2v)
    print('Processing text into tensors...')
    local f
    local vocab_count = {} -- vocab count 
    local idx2word = {'ZERO', 'START'} 
    local word2idx = {}; word2idx['ZERO'] = 1; word2idx['START'] = 2
    local split_count = 0
    local output_tensors1 = {}  --for sentence1
    local output_tensors2 = {}  -- for sentence2
    local labels = {}
    -- first go through train/valid/test to get max sentence length
    -- also counts the number of sentences
    f = io.open(test_file, 'r')       
    for line in f:lines() do
       line1 = line:gsub("^%%.*", "comment")
       if line1=="comment" or line == "" then
       else 
          split_count = split_count + 1
       end
    end
    f:close()
      
    print(string.format('(T,H) pair count: %d', split_count))
    -- Preallocate the tensors we will need.

    output_tensors1 = torch.ones(split_count, max_sentence_l):long() 
    output_tensors2 = torch.ones(split_count, max_sentence_l):long() 
    labels = torch.zeros(split_count):long() 

    f = io.open(test_file, 'r')
    local sentence_num = 0
    for line in f:lines() do
       line1 = line:gsub("^%%.*", "comment")
       if line1=="comment" or line == "" then
          --ignore comment and blank lines
       else 
          sentence_num = sentence_num + 1
          local datum = stringx.split(line, '\t')
          local _, s1, s2, label = datum[1], datum[2], datum[3], datum[4]
          if label ==nil or label =='-' then  
             -- skip these entries
          else
             labels[sentence_num] = ScorerBatchloader.labelToNumber(label) + 1
             -- append tokens in the sentence1
             output_tensors1[sentence_num][1] = 2 -- vj: 'START'
             local word_num = 1
             for rword in s1:gmatch'([^%s]+)' do
                word_num = word_num + 1
                if word2idx[rword]==nil then
                   idx2word[#idx2word + 1] = rword 
                   word2idx[rword] = #idx2word
                end
                output_tensors1[sentence_num][word_num] = word2idx[rword]
                if word_num == max_sentence_l then break end
             end
             -- append tokens in the sentence2
             output_tensors2[sentence_num][1] = 2
             word_num = 1
             for rword in s2:gmatch'([^%s]+)' do
                word_num = word_num + 1
                if word2idx[rword]==nil then
                   idx2word[#idx2word + 1] = rword 
                   word2idx[rword] = #idx2word
                end
                output_tensors2[sentence_num][word_num] = word2idx[rword]
                if word_num == max_sentence_l then break end
             end
          end
       end
    end
    -- vj: load up w2v[i] with the 300-long word-vector for the word whose index is i
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

    return output_tensors1, output_tensors2, labels, idx2word, word2idx, w2v
end

return ScorerBatchloader

