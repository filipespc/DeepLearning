# ----------------------------------------------------------------------------------------------
# -- MINIMIZE LOSS

training_op = tf.train.AdamOptimizer(0.001).minimize(loss)
