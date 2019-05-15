# ----------------------------------------------------------------------------------------------
# -- LAYERS

# Dense
dense_out = tf.layers.dense(prev_layer,num_outputs)

# Convolutional
conv_out = tf.layers.conv2d(prev_layer,num_filters,filter_size,strides=2,padding='same')

# ----------------------------------------------------------------------------------------------
# -- ACTIVATION FUNCTIONS

# Relu
relu_out = tf.maximum(alpha*prev_layer,prev_layer)

# sigmoid
sig_out = tf.sigmoid(prev_layer)

# ----------------------------------------------------------------------------------------------
# -- REGULARIZATION

# dropout
drop_out = tf.nn.dropout(prev_layer,keep_prob=dropout)

# ----------------------------------------------------------------------------------------------
# -- DATA MANIPULATION

# reshape - usually when the previous layer is a convolutional layer and the next is a dense layer
resh_out = tf.reshape(pre_layer, (-1,width*height*num_filters))
