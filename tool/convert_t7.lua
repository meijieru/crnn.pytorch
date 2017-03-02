require('table')
require('torch')
require('os')

function clone(t)
    -- deep-copy a table
    if type(t) ~= "table" then return t end
    local meta = getmetatable(t)
    local target = {}
    for k, v in pairs(t) do
        if type(v) == "table" then
            target[k] = clone(v)
        else
            target[k] = v
        end
    end
    setmetatable(target, meta)
    return target
end


function tableMerge(lhs, rhs)
    output = clone(lhs)
    for _, v in pairs(rhs) do
        table.insert(output, v)
    end
    return output
end


function isInTable(val, val_list)
    for _, item in pairs(val_list) do
        if val == item then
            return true
        end
    end
    return false
end


function modelToList(model)
    local ignoreList = {
        'nn.Copy',
        'nn.AddConstant',
        'nn.MulConstant',
        'nn.View',
        'nn.Transpose',
        'nn.SplitTable',
        'nn.SharedParallelTable',
        'nn.JoinTable',
    }
    local state = {}
    local param
    for i, layer in pairs(model.modules) do
        local typeName = torch.type(layer)
        if not isInTable(typeName, ignoreList) then
            if typeName == 'nn.Sequential' or typeName == 'nn.ConcatTable' then
                param = modelToList(layer)
            elseif typeName == 'cudnn.SpatialConvolution' or typeName == 'nn.SpatialConvolution' then
                param = layer:parameters()
            elseif typeName == 'cudnn.SpatialBatchNormalization' or typeName == 'nn.SpatialBatchNormalization' then
                param = layer:parameters()
                bn_vars = {layer.running_mean, layer.running_var}
                param = tableMerge(param, bn_vars)
            elseif typeName == 'nn.LstmLayer' then
                param =  layer:parameters()
            elseif typeName == 'nn.BiRnnJoin' then
                param =  layer:parameters()
            elseif typeName == 'cudnn.SpatialMaxPooling' or typeName == 'nn.SpatialMaxPooling' then
                param = {}
            elseif typeName == 'cudnn.ReLU' or typeName == 'nn.ReLU' then
                param = {}
            else
                print(string.format('Unknown class %s', typeName))
                os.exit(0)
            end
            table.insert(state, {typeName, param})
        else
            print(string.format('pass %s', typeName))
        end
    end
    return state
end


function saveModel(model, output_path)
    local state =  modelToList(model)
    torch.save(output_path, state)
end
