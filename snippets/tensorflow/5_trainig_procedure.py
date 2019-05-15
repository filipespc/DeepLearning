# ----------------------------------------------------------------------------------------------
# -- TRAINING PROCEDURE

# TODO: Build Model
# ---- get place holders
# ---- get model (defining layers)
# ---- get loss function(s)
# ---- get training operation(s)

epoch_count = num_epochs # number of epochs during training
batch_size = num_examples_per_batch # number of examples per batch
learning_rate = 0.001 # learning rate
print_every = print_period # defines the number of epochs trained befor printing parcial results
saver = tf.train.Saver() # used for saving the model later

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    steps=0
    for epoch_i in range(epoch_count):
        # get_batches is a function that return one batch of "batch_size" examples
        for batch_inputs,batch_targets in get_batches(batch_size):
            steps += 1
            
            feed = {input_:batch_inputs, targets_:batch_targets, learning_rate_:learning_rate}
            # training_op is the training operation
            batch_loss, _ = sess.run([loss, training_op], feed_dict=feed)
            
            if steps % print_every == 0:
                print('Epoch'+'{}'.format(epoch_i).rjust(4)+'/{}'.format(epoch_count).ljust(5),end=' ')
                print('Loss:'+'{:.3f}'.format(batch_loss).rjust(7),end=' ')
                
    saver.save(sess, './checkpoints/generator.ckpt')      