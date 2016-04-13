-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'CONLL.hdf5', 'data file')
cmd:option('-classifier', 'hmm', 'classifier to use')

-- Hyperparameters
cmd:option('-beta', 1, 'beta for F-Score')
cmd:option('-laplace', 1, 'added counts for laplace smoothing')
cmd:option('-batchsize', 20, 'Batches to train at once')
cmd:option('-eta', 0.00004, 'Training eta')
cmd:option('-epochs', 20, 'Epochs to train for')

-- ...
function hmm(inputs, targets)
	local initial = torch.Tensor(nclasses,1):fill(0)
	initial[targets[1]] = 1.0
	local transition = torch.Tensor(nclasses, nclasses):fill(opt.laplace)
	local emission = torch.Tensor(nfeatures, nclasses):fill(opt.laplace)

	--train transition and emission
	local statecount = torch.Tensor(nclasses, nclasses):fill(opt.laplace*nclasses)
	local emissioncount = torch.Tensor(nclasses, nfeatures):fill(opt.laplace*nfeatures)
	local prevstate = targets[1]
	for i=1, targets:size(1) do
		--count transitions
		transition[targets[i]][prevstate] = transition[targets[i]][prevstate] + 1
		prevstate = targets[i]
		--count emissions
		emission[inputs[i]][prevstate] = emission[inputs[i]][prevstate] + 1
		--update count matrices (for later division)
		statecount:narrow(2, prevstate, 1):add(1)
		emissioncount:narrow(2, prevstate, 1):add(1)
	end	
	transition:cdiv(statecount)
	emission:cdiv(emissioncount)

	return initial:log(), emission:log(), transition:log()
end

function score_hmm(observations, i)
    local observation_emission = emission[observations[i]]:view(nclasses, 1):expand(nclasses, nclasses)
    -- NOTE: allocates a new Tensor
    return observation_emission + transition
end

function reshapeTrainingData(inputs, targets)
	newInput = torch.Tensor(inputs:size(1)-1, 2)
	newInput:narrow(2,1,1):copy(inputs:narrow(1,2,inputs:size(1)-1))
	newInput:narrow(2,2,1):copy(targets:narrow(1,1,inputs:size(1)-1))

	newTargets = torch.Tensor(inputs:size(1)-1, 1)
	newTargets:narrow(2,1,1):copy(targets:narrow(1,2,inputs:size(1)-1))

	return newInput, newTargets
end

function trainNN(model, criterion, input, target, vinput, vtarget)  
   print(input:size(1), "size of the test set")
   --SGD after torch nn tutorial and https://github.com/torch/tutorials/blob/master/2_supervised/4_train.lua
   for i=1, opt.epochs do
   	  epochLoss = 0
   	  numBatches = 0
      --shuffle data
      shuffle = torch.randperm(input:size(1))
      --mini batches, yay
      for t=1, input:size(1), opt.batchsize do
         xlua.progress(t, input:size(1))

         local inputs = torch.Tensor(opt.batchsize, input:size(2))
         local targets = torch.Tensor(opt.batchsize)
         local k = 1
         for i = t,math.min(t+opt.batchsize-1,input:size(1)) do
            -- load new sample
            inputs[k] = input[shuffle[i]]
            targets[k] = target[shuffle[i]]
            k = k+1
         end
         k=k-1
         --in case the last batch is < batchsize
         if k < opt.batchsize then
           inputs = inputs:narrow(1, 1, k):clone()
           targets = targets:narrow(1, 1, k):clone()
         end
         --zero out
         model:zeroGradParameters()
         --predict and compute loss
         preds = model:forward(inputs)
         loss = criterion:forward(preds, targets)      
         dLdpreds = criterion:backward(preds, targets)
         model:backward(inputs, dLdpreds)
         model:updateParameters(opt.eta)
         epochLoss = epochLoss + loss
         numBatches = numBatches + 1
      end
      --predicting accuracy of epoch

      print("\nepoch " .. i .. ", loss: " .. epochLoss/numBatches/opt.batchsize)
	  
      _, yhat = model:forward(input):max(2)
      validationFScore = fscore(yhat:squeeze(), target:squeeze())
      print(validationFScore, "FScore on whole Training set")

      _, yhat = model:forward(vinput):max(2)
      validationFScore = fscore(yhat:squeeze(), vtarget:squeeze())
      print(validationFScore, "FScore on whole Validation set")

	end
   return model
end

function memm(inputs, targets, valid, valid_target)	
	--no using embeddings at this point!

	linLayer = nn.Linear(2,nclasses)
	softmaxLayer = nn.LogSoftMax()

	model = nn.Sequential()
	model:add(linLayer):add(softmaxLayer)

	criterion = nn.ClassNLLCriterion()
	trainNN(model, criterion, inputs, targets, valid, valid_target)
	return model
end

function score_memm(observations, i)
	local scores = torch.Tensor(nclasses, nclasses)
	for class=1, nclasses do
		scores:narrow(2,class,1):copy(model:forward(observations[i]))
	end
	return scores

end
	

function sperc(inputs, targets)

end

function viterbi(observations, logscore)
	--Viterbi adjusted from the section notebook!
    local n = observations:size(1)
    local max_table = torch.Tensor(n, nclasses)
    local backpointer_table = torch.Tensor(n, nclasses)

    -- first timestep
    -- the initial most likely paths are the initial state distribution
    -- NOTE: another unnecessary Tensor allocation here
    if opt.classifier == 'hmm' then
    	maxes, backpointers = (initial + emission[observations[1]]):max(2)
    else
    	maxes, backpointers = logscore(observations, 1):max(2)
    end
    max_table[1] = maxes

    -- remaining timesteps ("forwarding" the maxes)
    for i=2,n do
        -- precompute edge scores
        y = logscore(observations, i)
        scores = y + maxes:view(1, nclasses):expand(nclasses, nclasses)

        -- compute new maxes (NOTE: another unnecessary Tensor allocation here)
        maxes, backpointers = scores:max(2)

        -- record
        max_table[i] = maxes
        backpointer_table[i] = backpointers
    end

    -- follow backpointers to recover max path
    local classes = torch.Tensor(n)
    maxes, classes[n] = maxes:max(1)
    for i=n,2,-1 do
        classes[i-1] = backpointer_table[{i, classes[i]}]
    end

    return classes
end

function fscore(yhat, y)
	--fscore for one-dimensional vector
	local precision = 0
	local recall = 0
	for i=1, yhat:size(1) do
		--true predictions	
		if yhat[i] > 1 and yhat[i] < 8 and y[i] == yhat[i] then
			precision = precision + 1
		end
		--of targets, how many were predicted
		if y[i] > 1 and y[i] < 8 and y[i] == yhat[i] then
			recall = recall + 1
		end
	end
	local score = (opt.beta^2 + 1) * precision * recall / 
						(opt.beta^2 * precision + recall)
	return math.max(score, 0)
end

function sentenceFscore(inputs, targets, logscore)
	sentence_count = 0
	fscore_sum = 0

	begin_index = 1
	for index=2, inputs:size(1) do
		if inputs[index][1] == 2 then
			current_prediction = viterbi(inputs:narrow(1, begin_index, index-begin_index), logscore)
			current_fscore = fscore(current_prediction, targets:narrow(1, begin_index, index-begin_index):squeeze())
			begin_index = index
			sentence_count = sentence_count + 1
			fscore_sum = fscore_sum + current_fscore
		end
	end
	return fscore_sum/sentence_count
end


function main() 
	-- Parse input params
	opt = cmd:parse(arg)
	local f = hdf5.open(opt.datafile, 'r')
	nclasses = f:read('nclasses'):all():long()[1]
	nfeatures = f:read('nfeatures'):all():long()[1]

	train_input = f:read('train_input'):all()
	train_target = f:read('train_output'):all()

	valid_input = f:read('valid_input'):all()
	valid_target = f:read('valid_output'):all()

	test_input = f:read('test_input'):all()
	embeddings = f:read('embeddings'):all()

	--test the f-score function
	test_tensor = torch.Tensor(10):fill(2)
	assert(fscore(test_tensor, test_tensor), 10)

	-- Train.
	if opt.classifier == 'hmm' then
		initial, emission, transition = hmm(train_input, train_target)
		-- print(viterbi(valid_input:narrow(1,1,15), score_hmm))
		-- print(valid_target:narrow(1,1,15))
		-- first15score = fscore(viterbi(valid_input:narrow(1,1,15), score_hmm), valid_target:narrow(1,1,15))
		-- print(first15score)
		print ("Average Training F-Score is " .. sentenceFscore(train_input, train_target, score_hmm))
		print ("Average Validation F-Score is " .. sentenceFscore(valid_input, valid_target, score_hmm))
	elseif opt.classifier == 'memm' then
		train_input, train_target = reshapeTrainingData(train_input,train_target)
		valid_input, valid_target = reshapeTrainingData(valid_input,valid_target)
		model = memm(train_input, train_target, valid_input, valid_target)
		print ("Average Training F-Score is " .. sentenceFscore(train_input, train_target, score_memm))
		print ("Average Validation F-Score is " .. sentenceFscore(valid_input, valid_target, score_memm))

	end
	-- Test.
end

main()
