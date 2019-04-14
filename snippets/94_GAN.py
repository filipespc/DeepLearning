# ----------------------------------------------------------------------------------------------
# -- VARIABLE SCOPE

# The discriminator should be declared within the following.
# "reuse" should be set to:
# ---- False when defining the outputs used for the loss of the real images (the labels should be 1 - or something like 0.9 for smoothing)
# ---- True when defining the outputs used for the loss of the fake images (the labels should be 0 - or something like 0.1 for smoothing)
with tf.variable_scope('discriminator', reuse=False):

# The generator should be declared within the following.
# "reuse" should be set to:
# ---- False when defining the outputs used for the loss (the labels should be 1 - or something like 0.9 for smoothing)
# ---- True when using the network to generate images
with tf.variable_scope('generator', reuse=False):

# ----------------------------------------------------------------------------------------------
# -- BATCH NORMALIZATION

# "training" must be set to:
# ---- True if the batch_normalization is on the discrimator
# ---- True if the batch_normalization is on the generator and the network is being trained
# ---- False if the batch_normalization is on the generator and the network is being used to generate images
bn_out = tf.layers.batch_normalization(prev_layer, training=True)

# ----------------------------------------------------------------------------------------------
# -- LOSS FUCTIONS

# Defining the losses for the generator (gen_loss) and the discriminator (dis_loss)
label_smooth = 0.9
gen_model = generator(input_z, out_channel_dim)
dis_out_real, dis_logit_real = discriminator(input_real, reuse=False)
dis_out_fake, dis_logit_fake = discriminator(gen_model, reuse=True)

gen_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dis_out_fake)*label_smooth,logits=dis_logit_fake), name='gen_loss')
dis_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dis_out_real)*label_smooth,logits=dis_logit_real), name='diss_loss_real')
dis_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dis_out_fake),logits=dis_logit_fake), name='diss_loss_fake')

dis_loss = tf.add(dis_loss_real, dis_loss_fake, name='dis_loss')

# ----------------------------------------------------------------------------------------------
# -- OPTIMIZER

# Training operations - this code allows the two networks to be trained separately. The with also guarantees that the batch normalization
# is being applyed. During training procedure, both operations (dis_train_oper and gen_train_ope) should be executed.
train_vars = tf.trainable_variables()
dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]
gen_vars = [var for var in train_vars if var.name.startswith('generator')]

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    dis_train_oper = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1).minimize(d_loss, var_list=dis_vars)
    gen_train_oper = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1).minimize(g_loss, var_list=gen_vars)