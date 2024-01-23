function vectors = get_vectors_symbolic(params)
    % Same as get_vectors but with symbolic math in MATLAB

    % extract the dimension
    d = (1 + sqrt(1 + 8 * length(params))) / 2;
    d = int32(d);

    % initialize with the trivial case of no phase
    vectors = {sym(ones(d, 1))};

    % for each remaining vector add the phase
    param_index = 1;
    for i = 2:d
        vec_i = sym(ones(d, 1)); % initialize vec_i with ones
        eigenval_index = param_index; % assign the first param as eigenvalue phase
        vec_i(2) = exp(2 * pi * sym('i') * params(param_index));
        param_index = param_index + 1;
        for j = 3:d
            if param_index > length(params)
                break;
            end
            if mod(j, 2) == 0
                % j is even, so we add the phase
                vec_i(j) = exp(2 * pi * sym('i') * params(param_index));
                param_index = param_index + 1;
            else
                % j is odd, and we use the eigenvalue index for phase
                vec_i(j) = exp(2 * pi * sym('i') * (params(param_index) + params(eigenval_index)));
            end
        end
        vectors{end + 1} = vec_i;
    end
end
