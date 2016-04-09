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
cmd:option('-laplace', 0, 'added counts for laplace smoothing')

-- ...
function hmm(inputs, targets, valid_inputs, valid_targets, test)
	local initial = torch.Tensor(nclasses):fill(0)
	initial[targets[1]] = 1.0
	local transition = torch.Tensor(nclasses, nclasses):fill(0)
	local emission = torch.Tensor(nclasses, nfeatures):fill(opt.laplace)

	--train transition and emission
	local statecount = torch.Tensor(nclasses, nclasses):fill(0)
	local emissioncount = torch.Tensor(nclasses, nfeatures):fill(opt.laplace*nclasses)
	local prevstate = targets[1]
	for i=1, targets:size(1) do
		--count transitions
		transition[prevstate][targets[i]] = transition[prevstate][targets[i]] + 1
		prevstate = targets[i]
		--count emissions
		emission[prevstate][inputs[i]] = emission[prevstate][inputs[i]] + 1
		--update count matrices (for later division)
		statecount:narrow(2, prevstate, 1):add(1)
		emissioncount:narrow(2, inputs[i], 1):add(1)
	end	
	transition:cdiv(statecount)
	emission:cdiv(emissioncount)
	print(emission:narrow(2, 1, 10))
end

function memm(inputs, targets, valid_inputs, valid_targets, test)

end

function sperc(inputs, targets, valid_inputs, valid_targets, test)

end

function viterbi()

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
	return score
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
		hmm(train_input, train_target, valid_input, valid_target, test_input)
	end
	-- Test.
end

main()
