import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    with tf.name_scope("masked_coss_entropy"):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels, name="loss")
        mask = tf.cast(mask, dtype=tf.float32, name='mask')
        mask /= tf.reduce_mean(mask)
        loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    with tf.name_scope('masked_accuracy'):
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)#【mask除以mask的均值，如果mask是0，则仍为0；如果mask为1，则mask变为1/（mask的均值）】
        accuracy_all *= mask#【如果mask为0，则准确率为0，这样未被mask的数据不参于准确率计算；如果mask为1，则相当于准确率乘了1/（mask的均值）】
    return tf.reduce_mean(accuracy_all)
