using Random;


function holdOut(N::Int, P::Float64)

    @assert 0 <= P <= 1 "P must be a value between 0 and 1.";

    indexes = randperm(N)
    testSamples = round(Int, N * P)

    testIndexes = indexes[1:testSamples]
    trainIndexes = indexes[testSamples+1:end]

    @assert isempty(intersect(Set(testIndexes), Set(trainIndexes))) "The sets are not disjoint"
    @assert length(testIndexes)+length(trainIndexes) == N "The size of sets are not equal to N"


    return (trainIndexes, testIndexes)
end



function holdOut(N::Int, Pval::Float64, Ptest::Float64)
    
    @assert (Pval + Ptest) < 1.0 "Pval and Ptest sum can't be greater than 1";

    validationAndTestPercentage = Pval + Ptest

    (trainIndexes, validationAndTestIndexes) = holdOut(N, validationAndTestPercentage)

    validationAndTestSamples = length(validationAndTestIndexes)
    
    # Relative percentage of validation set respect of size of validationAndTestSamples
    if validationAndTestSamples > 0
        relativeValidationPercentage = Pval / validationAndTestPercentage
    else
        relativeValidationPercentage = 0
    end
    

    (temporalValidationIndexes, temporalTestIndexes) = holdOut(validationAndTestSamples, 1.0 - relativeValidationPercentage)
    
    validationIndexes = validationAndTestIndexes[temporalValidationIndexes]
    testIndexes = validationAndTestIndexes[temporalTestIndexes]

    @assert isempty(intersect(Set(validationIndexes), Set(trainIndexes), Set(testIndexes))) "The sets are not disjoint"
    @assert length(validationIndexes)+length(trainIndexes)+length(testIndexes) == N "he size of sets are not equal to N"


    return (trainIndexes, validationIndexes, testIndexes)
end


function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
minValues = normalizationParameters[1];
maxValues = normalizationParameters[2];
dataset .-= minValues;
dataset ./= (maxValues .- minValues);
# eliminate any atribute that do not add information
dataset[:, vec(minValues.==maxValues)] .= 0;
return dataset;
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset , calculateMinMaxNormalizationParameters(dataset));
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
normalizeMinMax!(copy(dataset), normalizationParameters);
end;

function normalizeMinMax( dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));
end;



# Accuracy

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    mean(outputs.==targets);
end;


function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};      
    threshold::Real=0.5)
accuracy(outputs.>=threshold, targets);
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
    threshold::Real=0.5)

    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets);
    end;
end;

# CROSS validation

function crossvalidation(N::Int64, k::Int64)

    vector=collect(1:k)

    number_repetitions = ceil(Int, N/k)

    vector = repeat(vector, number_repetitions)

    return shuffle!(vector[1:N])
end



function crossvalidation(targets::AbstractArray{Bool, 1}, k::Int64)

    indices_vector = zeros(Int, size(targets,1))

    positive_indexes = findall(t -> t, targets)
    negative_indexes = findall(t -> !t, targets)

    indices_vector[positive_indexes] = crossvalidation(size(positive_indexes,1), k)
    indices_vector[negative_indexes] = crossvalidation(size(negative_indexes,1), k)

    return indices_vector;

end


function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)

  indices_vector = zeros(Int, size(targets,1))

  [indices_vector[findall(targets[:,i])] = crossvalidation(sum(targets[:, i]), k) for i in 1:size(targets,2)]

  return indices_vector

end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    result =  crossvalidation(oneHotEncoding(targets), k)
    @assert size(targets) == size(result)
    return result;
  end
