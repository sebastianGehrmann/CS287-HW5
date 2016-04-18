-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'CONLL.hdf5', 'data file')
cmd:option('-classifier', 'hmm', 'classifier to use')
cmd:option('-outfile', '', 'output file')
cmd:option('-use_all', 0, 'Use all data')

-- Hyperparameters
cmd:option('-beta', 1, 'beta for F-Score')
cmd:option('-laplace', 1, 'added counts for laplace smoothing')
cmd:option('-batchsize', 16, 'Batches to train at once')
cmd:option('-eta', 0.005, 'Training eta')
cmd:option('-epochs', 40, 'Epochs to train for')
cmd:option('-nwindow', 7, 'Window size')
cmd:option('-nembed', 50, 'Embedding layer')
cmd:option('-nembed2', 20, 'Embedding layer for class')
cmd:option('-nhidden', 100, 'Hidden layer')
cmd:option('-use_embedding', 1, 'Use embedding')
cmd:option('-use_aux', 1, 'Use auxillary features')
cmd:option('-use_structured', 0, 'Use structured perceptron')
cmd:option('-use_averaging', 1, 'Use averaging in structured perceptron')

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
    local emission = torch.Tensor(nwords, nclasses):fill(opt.laplace)

    --train transition and emission
    local statecount = torch.Tensor(nclasses, nclasses):fill(opt.laplace*nclasses)
    local emissioncount = torch.Tensor(nclasses, nwords):fill(opt.laplace*nwords)
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
        newInput = torch.cat(
            inputs:narrow(1, 2, inputs:size(1)-1),
            targets:narrow(1, 1, inputs:size(1)-1):view(inputs:size(1)-1, 1)
        )
        newTargets = targets:narrow(1, 2, inputs:size(1)-1):clone()
    else
        newInput = torch.cat(inputs, torch.Tensor(inputs:size(1), 1):fill(0))
    end

    return newInput, newTargets
end


function trainNN(model, criterion, input, target, vinput, vtarget)
    print("input size", #input)
    print("eta", opt.eta)
    print("epochs", opt.epochs)
    -- SGD after torch nn tutorial and https://github.com/torch/tutorials/blob/master/2_supervised/4_train.lua
    for iter = 1, opt.epochs do
        epochLoss = 0
        -- shuffle data
        shuffle = torch.randperm(input:size(1))
        -- mini batches, yay
        for t=1, input:size(1), opt.batchsize do
            xlua.progress(t, input:size(1))

            local inputs = torch.Tensor(opt.batchsize, input:size(2))
            local targets = torch.Tensor(opt.batchsize)
            local k = 1
            for i = t, math.min(t+opt.batchsize-1,input:size(1)) do
                -- load new sample
                inputs[k] = input[shuffle[i]]
                targets[k] = target[shuffle[i]]
                k = k+1
            end
            k = k-1
            -- in case the last batch is < batchsize
            if k < opt.batchsize then
                inputs = inputs:narrow(1, 1, k):clone()
                targets = targets:narrow(1, 1, k):clone()
            end
            -- zero out
            model:zeroGradParameters()
            -- predict and compute loss
            preds = model:forward(inputs)
            loss = criterion:forward(preds, targets)

            dLdpreds = criterion:backward(preds, targets)
            -- Hack: remove concat layer and then add it back, since LookupTable doesn't return valid gradients
            local concatLayer = model:get(1)
            model:remove(1)
            model:backward(concatLayer:forward(inputs), dLdpreds)
            model:updateParameters(opt.eta)
            model:insert(concatLayer, 1)

            epochLoss = epochLoss + loss
        end
        -- predicting accuracy of epoch

        print("\nepoch " .. iter .. ", loss: " .. epochLoss / input:size(1))

        _, yhat = model:forward(input):max(2)
        validationFScore = fscore(yhat:squeeze(), target:squeeze())
        print(validationFScore, "FScore on whole Training set")

        _, yhat = model:forward(vinput):max(2)
        validationFScore = fscore(yhat:squeeze(), vtarget:squeeze())
        print(validationFScore, "FScore on whole Validation set")
    end
    return model
end


function memm(inputs)
    -- no using embeddings at this point!
    local model = nn.Sequential()

    local concat = nn.ConcatTable()
    concat:add(nn.Narrow(2, 1, nwindow))
    local naux = inputs:size(2)-nwindow-1
    if use_aux > 0 then
        concat:add(nn.Narrow(2, nwindow+1, naux))
    end
    concat:add(nn.Select(2, inputs:size(2)))
    model:add(concat)

    local parallel = nn.ParallelTable()
    -- words features
    local wordtable = nn.Sequential()
    local wordlookup = nn.LookupTable(nwords, nclasses)
    wordtable:add(wordlookup)
    wordtable:add(nn.Sum(1, 2))
    parallel:add(wordtable)
    -- aux features
    if use_aux > 0 then
        parallel:add(nn.Linear(naux, nclasses))
    end
    -- prev class features
    local classlookup = nn.LookupTable(nclasses, nclasses)
    parallel:add(classlookup)
    model:add(parallel)
    
    -- Sum over the 3 concatenated tables (linear model)
    model:add(nn.JoinTable(1, 1))
    model:add(nn.View(2+use_aux, nclasses))
    model:add(nn.Sum(1, 2))
    return model
end


function memm2(inputs)
    -- using embeddings
    local model = nn.Sequential()

    local concat = nn.ConcatTable()
    concat:add(nn.Narrow(2, 1, nwindow))
    local naux = inputs:size(2)-nwindow-1
    if use_aux > 0 then
        concat:add(nn.Narrow(2, nwindow+1, naux))
    end
    concat:add(nn.Select(2, inputs:size(2)))
    model:add(concat)

    local parallel = nn.ParallelTable()
    -- words features
    local wordtable = nn.Sequential()
    local wordlookup = nn.LookupTable(nwords, nembed)
    if use_embedding > 0 then
        wordlookup.weight:copy(embeddings)
    end 
    wordtable:add(wordlookup)
    wordtable:add(nn.View(nwindow * nembed):setNumInputDims(2))
    parallel:add(wordtable)
    -- aux features
    if use_aux > 0 then
        parallel:add(nn.Identity())
    end
    -- prev class features
    local classlookup = nn.LookupTable(nclasses, nembed2)
    parallel:add(classlookup)
    model:add(parallel)
    
    -- Linear model over the concatenated features
    model:add(nn.JoinTable(1, 1))
    model:add(nn.Linear(nwindow*nembed + naux*use_aux + nembed2, nhidden))
    model:add(nn.HardTanh())
    model:add(nn.Linear(nhidden, nclasses))
    return model
end


function trainNNClassifier(model_fn, inputs, targets, valid, valid_target)
    local model = model_fn(inputs)
    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion(nil, false)
    trainNN(model, criterion, inputs, targets, valid, valid_target)
    return model
end


function score_memm(model, cur)
    cur = cur:clone()
    local scores = torch.Tensor(nclasses, nclasses)
    for class=1, nclasses do
        cur[cur:size(1)] = class
        local res = model:forward(cur:view(1, cur:size(1)))
        scores:select(2, class):copy(res)
    end
    return scores
end


function viterbi(observations, score_fn)
    local n = observations:size(1)
    assert(observations[1][1] == startword)
    assert(observations[n][1] == endword)

    --Viterbi adjusted from the section notebook!
    local max_table = torch.Tensor(n, nclasses)
    local backpointer_table = torch.Tensor(n, nclasses)

    -- first timestep
    -- the initial most likely paths are the initial state distribution
    if opt.classifier == 'hmm' then
        maxes = initial + emission[observations[1][1]]
    else
        -- Force first tag to be start tag
        maxes = initial + score_fn(observations[1]):max(2)
    end
    max_table[1] = maxes
    -- backpointers are meaningless here

    -- remaining timesteps ("forwarding" the maxes)
    for i=2,n do
        -- precompute edge scores
        y = score_fn(observations[i])
        scores = y + maxes:view(1, nclasses):expand(nclasses, nclasses)

        -- compute new maxes
        maxes, backpointers = scores:max(2)
        -- Force intermediate tags to not be start/end tags
        maxes[startclass] = -math.huge
        maxes[endclass] = -math.huge

        -- record
        max_table[i] = maxes
        backpointer_table[i] = backpointers
    end

    -- follow backpointers to recover max path
    local classes = torch.Tensor(n)
    -- Force last tag to be end tag
    classes[n] = endclass
    for i=n, 2, -1 do
        classes[i-1] = backpointer_table[i][classes[i]]
    end
    return classes
end


function trainNNViterbi(model, input, target, vinput, vtarget)
    print("input size", #input)
    print("eta", opt.eta)
    print("epochs", opt.epochs)
    local score_fn = function(cur) return score_memm(model, cur) end
    local params, gradParams = model:getParameters()
    params:zero()
    local totalparams = torch.Tensor(params:size()):zero()
    local totalcnt = 0
    for iter = 1, opt.epochs do
        epochLoss = 0

        begin_index = nil
        for index = 1, input:size(1) do
            -- Sentence delimiter
            if input[index][1] == startword then
                begin_index = index
            elseif begin_index and input[index][1] == endword then
                xlua.progress(index, input:size(1))

                inputs = input:narrow(1, begin_index, index-begin_index+1):clone()
                targets = target:narrow(1, begin_index, index-begin_index+1):clone()
                prediction = viterbi(inputs, score_fn)

                -- Hack: remove concat layer and then add it back, since LookupTable doesn't return valid gradients
                local concatLayer = model:get(1)
                model:remove(1)
                -- zero out
                model:zeroGradParameters()
                -- Iterate through, tracking prev predicted class
                local prevclass = startclass
                for i = 1, inputs:size(1) do
                    if prediction[i] ~= targets[i] then
                        local cur = inputs[i]:clone()
                        cur[cur:size(1)] = prevclass
                        -- Pass through first layer manually
                        local table = concatLayer:forward(cur:view(1, cur:size(1)))

                        local preds = model:forward(table)
                        -- Manually compute gradients
                        local dLdpreds = torch.Tensor(1, nclasses):zero()
                        dLdpreds[1][targets[i]] = -1
                        dLdpreds[1][prediction[i]] = 1
                        model:backward(table, dLdpreds)
                    end
                    prevclass = prediction[i]
                end
                model:updateParameters(opt.eta)
                model:insert(concatLayer, 1)

                totalparams = totalparams + params
                totalcnt = totalcnt + 1
            end
        end

        _, yhat = model:forward(input):max(2)
        validationFScore = fscore(yhat:squeeze(), target:squeeze())
        print(validationFScore, "FScore on whole Training set")

        _, yhat = model:forward(vinput):max(2)
        validationFScore = fscore(yhat:squeeze(), vtarget:squeeze())
        print(validationFScore, "FScore on whole Validation set")
    end

    if opt.use_averaging > 0 then
        print("averaging over", totalcnt)
        params:copy(totalparams / totalcnt)
    end
end


function trainNNStructured(model_fn, inputs, targets, valid, valid_target)
    local model = model_fn(inputs)

    trainNNViterbi(model, inputs, targets, valid, valid_target)
    return model
end


function fscore(yhat, y)
    --fscore for one-dimensional vector
    local precision = 0
    local precisioncnt = 0
    local recall = 0
    local recallcnt = 0
    for i=1, yhat:size(1) do
        --true predictions
        if yhat[i] > 1 and yhat[i] < startclass then
            if y[i] == yhat[i] then
                precision = precision + 1
            end
            precisioncnt = precisioncnt + 1
        end
        -- of targets, how many were predicted
        if y[i] > 1 and y[i] < startclass then
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
        -- Sentence delimiter
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
        -- Sentence delimiter
        if inputs[index][1] == startword then
            begin_index = index
        elseif begin_index and inputs[index][1] == endword then
            prediction = viterbi(inputs:narrow(1, begin_index, index-begin_index+1), logscore)

            f:write(curind, ",")
            prev = false
            prevsuff = nil
            for i = 1, prediction:size(1) do
                cur = prediction[i]
                if cur > 1 and cur < startclass then
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
    nwords = f:read('nwords'):all():long()[1]

    nwindow = opt.nwindow
    nembed = opt.nembed
    nembed2 = opt.nembed2
    nhidden = opt.nhidden
    use_embedding = opt.use_embedding
    use_aux = opt.use_aux
    use_structured = opt.use_structured

    -- This is the first word
    startword = 2
    endword = 3
    -- This is the tag for <s>
    startclass = nclasses - 1
    -- This is the tag for </s>
    endclass = nclasses
    -- Initial state for HMM
    initial = torch.Tensor(nclasses, 1):fill(0)
    initial[startclass] = 1.0
    initial:log()

    print("nwords", nwords)
    print("nclasses", nclasses)
    print("startclass", startclass)
    print("endclass", endclass)

    train_input = f:read('train_input'):all():double()
    nfeatures = train_input:size(2)
    print("nfeatures", nfeatures)
    train_target = f:read('train_output'):all():double()
    train_input, train_target = reshapeTrainingData(train_input, train_target)

    valid_input = f:read('valid_input'):all():double()
    valid_target = f:read('valid_output'):all():double()
    valid_input, valid_target = reshapeTrainingData(valid_input, valid_target)

    if opt.use_all > 0 then
        train_input = torch.cat(train_input, valid_input, 1)
        train_target = torch.cat(train_target, valid_target, 1)
    end

    test_input = f:read('test_input'):all():double()
    -- For consistency
    test_input = reshapeTrainingData(test_input, nil)

    embeddings = f:read('embeddings'):all()

    print("train size", #train_input)
    print("valid size", #valid_input)
    print("test size", #test_input)

    --test the f-score function
    test_tensor = torch.Tensor(10):fill(2)
    assert(fscore(test_tensor, test_tensor), 10)

    -- Train.
    local score_fn = nil
    if classifier == 'hmm' then
        emission, transition = hmm(train_input, train_target)
        score_fn = score_hmm
    else
        local model_fn = nil
        if classifier == 'memm' then
            model_fn = memm
        elseif classifier == 'memm2' then
            model_fn = memm2
        end
        if use_structured > 0 then
            model = trainNNStructured(model_fn, train_input, train_target, valid_input, valid_target)
        else
            model = trainNNClassifier(model_fn, train_input, train_target, valid_input, valid_target)
        end
        score_fn = function(cur) return score_memm(model, cur) end
    end
    print("Average Training F-Score is " .. sentenceFscore(train_input, train_target, score_fn))
    print("Average Validation F-Score is " .. sentenceFscore(valid_input, valid_target, score_fn))

    -- Test.
    if outfile and outfile:len() > 0 then
        print("Writing to", outfile)
        local f_preds = io.open(outfile, "w")
        print_pred_labels(f_preds, test_input, score_fn)
    end
    print("Done!")
end

main()
