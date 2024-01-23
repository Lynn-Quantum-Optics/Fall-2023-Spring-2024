 function isInvertible = is_matrix_invertible(symMatrix)
    % This function checks if the symbolic matrix is invertible.
    
    % Ensure the matrix is square
    [rows, cols] = size(symMatrix);
    if rows ~= cols
        error('The matrix must be square to be invertible.');
    end
    
    % Calculate the determinant
    detVal = det(symMatrix);
    disp('determinant');
    disp(detVal);
    
    % Check if the determinant is not zero (which would indicate invertibility)
    isInvertible = ~isequal(detVal, sym(0));
end
