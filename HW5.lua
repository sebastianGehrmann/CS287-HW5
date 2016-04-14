-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'CONLL.hdf5', 'data file')
cmd:option('-classifier', 'hmm', 'classifier to use')
cmd:option('-outfile', '', 'output file')

-- Hyperparameters
cmd:option('-beta', 1, 'beta for F-Score')
cmd:option('-laplace', 1, 'added counts for laplace smoothing')
cmd:option('-batchsize', 16, 'Batches to train at once')
cmd:option('-eta', 0.0005, 'Training eta')
cmd:option('-epochs', 40, 'Epochs to train for')
cmd:option('-nembed', 50, 'Embedding layer')
cmd:option('-nhidden', 100, 'Hidden layer')
cmd:option('-use_embedding', 1, 'Use embedding')

label_map = {
    [2] = "I-PER",
    [3] = "I-LOC",
    [4] = "I-ORG",
    [5] = "I-MISC",
    [6] = "B-MISC",
    [7] = "B-LOC"
}

-- ...
function hmm(inputs, targets)
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
        emission[inputs[i][1]][prevstate] = emission[inputs[i][1]][prevstate] + 1
        --update count matrices (for later division)
        statecount:narrow(2, prevstate, 1):add(1)
        emissioncount:narrow(2, prevstate, 1):add(1)
    end
    transition:cdiv(statecount)
    emission:cdiv(emissioncount)

    return emission:log(), transition:log()
end

function score_hmm(cur)
    local observation_emission = emission[cur[1]]:view(nclasses, 1):expand(nclasses, nclasses)
    -- NOTE: allocates a new Tensor
    return observation_emission + transition
end

function reshapeTrainingData(inputs, targets)
    if targets then
        newInput = torch.Tensor(inputs:size(1)-1, 2):long():fill(0)
        newInput:narrow(2,1,1):copy(inputs:narrow(1,2,inputs:size(1)-1))

        newInput:narrow(2,2,1):copy(targets:narrow(1,1,inputs:size(1)-1))

        newTargets = torch.Tensor(inputs:size(1)-1):long()
        newTargets:copy(targets:narrow(1,2,inputs:size(1)-1))
    else
        newInput = torch.Tensor(inputs:size(1), 2):long():fill(0)
        newInput:narrow(2,1,1):copy(inputs)
    end

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

            -- Hack: the splitting layer must be removed before calling
            -- backward or else nn will try to propagate some empty
            -- gradients (since lookuptables have no gradients)
            local splitLayer = model:get(1)
            model:remove(1)
            dLdpreds = criterion:backward(preds, targets)
            model:backward(splitLayer:forward(inputs), dLdpreds)
            model:updateParameters(opt.eta)
            -- Add back
            model:insert(splitLayer, 1)

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
    local model = nn.Sequential()
    model:add(nn.SplitTable(1, 1))

    local parallel = nn.ParallelTable()

    -- words features
    local wordtable = nn.Sequential()
    -- Sparse feaure logistic regression
    local wordlookup = nn.LookupTable(nfeatures, nclasses)
    wordtable:add(wordlookup)
    parallel:add(wordtable)

    -- prev class features
    local classtable = nn.Sequential()
    -- Sparse feature logistic regression
    local classlookup = nn.LookupTable(nclasses, nclasses)
    classtable:add(classlookup)
    parallel:add(classtable)

    model:add(parallel)
    
    -- Join over last dimension
    model:add(nn.JoinTable(1, 1))
    -- Sum over the joined tables (linear model)
    model:add(nn.View(2, nclasses))
    model:add(nn.Sum(1, 2))

    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()
    trainNN(model, criterion, inputs, targets, valid, valid_target)
    return model
end

function memm2(inputs, targets, valid, valid_target)
    -- using embeddings
    local model = nn.Sequential()
    model:add(nn.SplitTable(1, 1))

    local parallel = nn.ParallelTable()
    -- words features
    local wordtable = nn.Sequential()
    local wordlookup = nn.LookupTable(nfeatures, nembed)
    if use_embedding > 0 then
        wordlookup.weight:copy(embeddings)
    end 
    wordtable:add(wordlookup)
    parallel:add(wordtable)

    -- class features
    local classtable = nn.Sequential()
    local classlookup = nn.LookupTable(nclasses, nembed)
    classtable:add(classlookup)
    parallel:add(classtable)
    model:add(parallel)
    
    -- Join over last dimension
    model:add(nn.JoinTable(1, 1))
    model:add(nn.Linear(nembed+nembed, nhidden))
    model:add(nn.HardTanh())
    model:add(nn.Linear(nhidden, nclasses))
    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()
    trainNN(model, criterion, inputs, targets, valid, valid_target)
    return model
end

function score_memm(cur)
    local scores = torch.Tensor(nclasses, nclasses)
    for class=1, nclasses do
        cur[2] = class
        local res = model:forward(torch.Tensor(1, 2):copy(cur))
        scores:select(2, class):copy(res)
    end
    return scores
end


function sperc(inputs, targets)

end

function viterbi(observations, logscore)
    assert(observations[1][1] == startword)

    --Viterbi adjusted from the section notebook!
    local n = observations:size(1)
    local max_table = torch.Tensor(n, nclasses)
    local backpointer_table = torch.Tensor(n, nclasses)

    -- first timestep
    -- the initial most likely paths are the initial state distribution
    -- NOTE: another unnecessary Tensor allocation here
    if opt.classifier == 'hmm' then
        maxes = initial + emission[observations[1][1]]
    else
        -- Force it to start at <s>, even though the NN should've figured this out
        maxes = initial + logscore(observations[1]):max(2)
    end
    max_table[1] = maxes
    -- backpointers are meaningless here

    -- remaining timesteps ("forwarding" the maxes)
    for i=2,n do
        -- precompute edge scores
        y = logscore(observations[i])
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
        classes[i-1] = backpointer_table[i][classes[i]]
    end

    return classes
end

function fscore(yhat, y)
    --fscore for one-dimensional vector
    local precision = 0
    local precisioncnt = 0
    local recall = 0
    local recallcnt = 0
    for i=1, yhat:size(1) do
        --true predictions
        if yhat[i] > 1 and yhat[i] < initclass then
            if y[i] == yhat[i] then
                precision = precision + 1
            end
            precisioncnt = precisioncnt + 1
        end
        -- of targets, how many were predicted
        if y[i] > 1 and y[i] < initclass then
            if y[i] == yhat[i] then
                recall = recall + 1
            end
            recallcnt = recallcnt + 1
        end
    end
    if recallcnt == 0 and precisioncnt == 0 then
        return 1
    end
    if recallcnt == 0 or precisioncnt == 0 then
        return 0
    end
    precision = precision / precisioncnt
    recall = recall / recallcnt 
    local score = ((opt.beta^2 + 1) * precision * recall /
                   (opt.beta^2 * precision + recall))
    return math.max(score, 0)
end

function sentenceFscore(inputs, targets, logscore)
    sentence_count = 0
    fscore_sum = 0

    begin_index = nil
    for index=1, inputs:size(1) do
        -- Sentence delimeter
        if inputs[index][1] == startword then
            begin_index = index
        elseif begin_index and inputs[index][1] == endword then
            current_prediction = viterbi(inputs:narrow(1, begin_index, index-begin_index+1), logscore)
            current_fscore = fscore(current_prediction, targets:narrow(1, begin_index, index-begin_index+1):squeeze())
            sentence_count = sentence_count + 1
            fscore_sum = fscore_sum + current_fscore
        end
    end
    return fscore_sum / sentence_count
end


function print_pred_labels(f, inputs, logscore)
    f:write("ID,Labels\n")

    begin_index = nil
    curind = 1
    for index=1, inputs:size(1) do
        -- Sentence delimeter
        if inputs[index][1] == startword then
            begin_index = index
        elseif begin_index and inputs[index][1] == endword then
            prediction = viterbi(inputs:narrow(1, begin_index, index-begin_index+1), logscore)

            f:write(curind, ",")
            prev = false
            prevsuff = nil
            for i = 1, prediction:size(1) do
                cur = prediction[i]
                if cur > 1 and cur < initclass then
                    str = label_map[cur]
                    suff = str:sub(3)
                    if prevsuff and prevsuff == suff and str:sub(1, 1) == "I" then
                        f:write("-", i-1)
                    else
                        if prev then
                            f:write(" ")
                        end
                        f:write(suff, "-", i-1)
                    end
                    prevsuff = suff
                    prev = true
                else
                    prevsuff = nil
                end
            end
            curind = curind + 1
            f:write("\n")
        end
    end
end


function main()
    -- Parse input params
    opt = cmd:parse(arg)
    local f = hdf5.open(opt.datafile, 'r')
    classifier = opt.classifier
    outfile = opt.outfile

    nclasses = f:read('nclasses'):all():long()[1]
    nfeatures = f:read('nfeatures'):all():long()[1]

    nembed = opt.nembed
    nhidden = opt.nhidden
    use_embedding = opt.use_embedding

    -- This is the first word
    startword = 2
    endword = 3
    -- This is the tag for <s>
    initclass = 8
    initial = torch.Tensor(nclasses,1):fill(0)
    initial[initclass] = 1.0
    initial:log()

    print("nfeatures", nfeatures)
    print("nclasses", nclasses)
    print("initclass", initclass)

    train_input = f:read('train_input'):all():long()
    train_target = f:read('train_output'):all():long()
    train_input, train_target = reshapeTrainingData(train_input, train_target)

    valid_input = f:read('valid_input'):all():long()
    valid_target = f:read('valid_output'):all():long()
    valid_input, valid_target = reshapeTrainingData(valid_input, valid_target)

    test_input = f:read('test_input'):all():long()
    -- For consistency
    test_input = reshapeTrainingData(test_input, nil)

    embeddings = f:read('embeddings'):all()

    print("train size", train_input:size())
    print("valid size", valid_input:size())
    print("test size", test_input:size())

    --test the f-score function
    test_tensor = torch.Tensor(10):fill(2)
    assert(fscore(test_tensor, test_tensor), 10)

    -- Train.
    if classifier == 'hmm' then
        emission, transition = hmm(train_input, train_target)
        -- print(viterbi(valid_input:narrow(1,1,15), score_hmm))
        -- print(valid_target:narrow(1,1,15))
        -- first15score = fscore(viterbi(valid_input:narrow(1,1,15), score_hmm), valid_target:narrow(1,1,15))
        -- print(first15score)
        print ("Average Training F-Score is " .. sentenceFscore(train_input, train_target, score_hmm))
        print ("Average Validation F-Score is " .. sentenceFscore(valid_input, valid_target, score_hmm))
    elseif classifier == 'memm' then
        model = memm(train_input, train_target, valid_input, valid_target)
        print ("Average Training F-Score is " .. sentenceFscore(train_input, train_target, score_memm))
        print ("Average Validation F-Score is " .. sentenceFscore(valid_input, valid_target, score_memm))
    elseif classifier == 'memm2' then
        model = memm2(train_input, train_target, valid_input, valid_target)
        print ("Average Training F-Score is " .. sentenceFscore(train_input, train_target, score_memm))
        print ("Average Validation F-Score is " .. sentenceFscore(valid_input, valid_target, score_memm))
    end

    -- Test.
    if outfile and outfile:len() > 0 then
        print("Writing to", outfile)
        local f_preds = io.open(outfile, "w")
        local fn = score_hmm
        if classifier ~= "hmm" then
            fn = score_memm
        end
        print_pred_labels(f_preds, test_input, fn)
    end
    print("Done!")
end

main()
