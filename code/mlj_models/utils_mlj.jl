function prepare_data_for_mlj(input_data::Matrix, output_data::CategoricalArray, train_split_ratio::Float64)
    
    input_data = normalizeMinMax!(convert(Array{Float32,2}, input_data[:,1:end]))
    datasetLength = size(input_data, 1)
    (trainIndexes, testIndexes) = holdOut(datasetLength, train_split_ratio)

    train_input_no_coerce, train_output_no_coerce = (MLJ.table(input_data[trainIndexes, :]), output_data[trainIndexes])
    test_input_no_coerce, test_output_no_coerce = (MLJ.table(input_data[testIndexes, :]), output_data[testIndexes])
  
   train_input, train_output = (coerce(train_input_no_coerce, autotype(train_input_no_coerce, rules=(:discrete_to_continuous,))), categorical(train_output_no_coerce))
   test_input, test_output = (coerce(test_input_no_coerce, autotype(test_input_no_coerce, rules=(:discrete_to_continuous,))), categorical(test_output_no_coerce))

    return (train_input, train_output, test_input, test_output)
end