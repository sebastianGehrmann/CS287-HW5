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

function memm(inputs, targets)

end

function sperc(inputs, targets)

end

function viterbi(observations, logscore)
    local n = observations:size(1)
    local max_table = torch.Tensor(n, nclasses)
    local backpointer_table = torch.Tensor(n, nclasses)

    -- first timestep
    -- the initial most likely paths are the initial state distribution
    -- NOTE: another unnecessary Tensor allocation here
    local maxes, backpointers = (initial + emission[observations[1]]):max(2)
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
		initial, emission, transition = hmm(train_input, train_target)
		-- print(viterbi(valid_input:narrow(1,1,15), score_hmm))
		-- print(valid_target:narrow(1,1,15))
		first15score = fscore(viterbi(valid_input:narrow(1,1,15), score_hmm), valid_target:narrow(1,1,15))
		print(first15score)

	end
	-- Test.
end

main()
