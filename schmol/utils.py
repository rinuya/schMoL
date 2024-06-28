def vector_subtraction(vec_a, vec_b):
    assert len(vec_a) == len(vec_b), "Vector subtraction only possible with same sized vectors" 
    return [a-b for a, b in zip(vec_a, vec_b)]

# returns loss of this vector
def vector_loss(vec_a, vec_b):
    loss_vector = vector_subtraction(vec_a, vec_b)
    return sum(loss_vector)