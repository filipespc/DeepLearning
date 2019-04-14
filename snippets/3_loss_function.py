# ----------------------------------------------------------------------------------------------
# -- SIGMOID CROSS ENTROPY

loss = df.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))