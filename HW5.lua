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

-- ...


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

   train_input = f:read('train_input')
   train_target = f:read('train_output')

   valid_input = f:read('valid_input')
   valid_target = f:read('valid_output')

   test_input = f:read('test_input')
   embeddings = f:read('embeddings') 

   --test the f-score function
   test_tensor = torch.Tensor(10):fill(2)
   assert(fscore(test_tensor, test_tensor), 10)
   
   -- Train.

   -- Test.
end

main()
