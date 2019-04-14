def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    input_ph = tf.placeholder(tf.int32,(None,None), name='input')
    target_ph = tf.placeholder(tf.int32,(None,None), name='target')
    lr_ph = tf.placeholder(tf.float32, name='learning_rate')
    # TODO: Implement Function
    return input_ph, target_ph, lr_ph