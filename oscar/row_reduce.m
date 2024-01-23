function rref_matrix = row_reduce(d)
    % Computes the row reduced form of the matrix of vectors to test linear independence

    % get the params
    syms p [1 (d - 1) * (floor(d / 2))] real;

    % get vectors
    vectors = get_vectors_symbolic(p);

    % create a matrix where the column vectors are the vectors
    result_matrix = [];

    % Concatenate each matrix in the list as a new column
    for i = 1:length(vectors)
        result_matrix = [result_matrix, vectors{i}];
    end

    rref_matrix = rref(result_matrix);
end
