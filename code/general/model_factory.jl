## Model cross validation
include("../mlj_models/train_mlj.jl")
include("../ann/build_train.jl")

  function modelCrossValidation(
    modelType::Symbol, modelHyperparameters::Dict,
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1})


    if(modelType == :ANN)


        topology = get(modelHyperparameters, :topology, [4,3])
        transferFunctions = get(modelHyperparameters, :transferFunctions, fill(σ, length(topology)))
        learningRate = get(modelHyperparameters, :learningRate, 0.01)
        maxEpochs = get(modelHyperparameters, :maxEpochs, 1000)
        validationRatio = get(modelHyperparameters, :validationRatio, 0)
        maxEpochsVal = get(modelHyperparameters, :maxEpochsVal, 20)

    
        return ANNCrossValidation(topology, dataset, crossValidationIndices,
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs, 
        learningRate=learningRate,
        validationRatio=validationRatio, 
        maxEpochsVal=maxEpochsVal)


    elseif modelType in [:SVC, :DecisionTreeClassifier, :KNeighborsClassifier]
        # Los modelos de MLJ son deterministas, usamos la nueva función
        return mljCrossValidation(modelType, modelHyperparameters, dataset, crossValidationIndices)
    else
        error("Tipo de modelo desconocido: ", modelType)
    end

end
