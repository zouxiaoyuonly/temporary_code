import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import data_handler

w2v = False
maxSeqLength = 90 
batch_size = 512
embedding_size = 512
hidden_units = 50
vocab_size = len(data_handler.word_index_dict) + 1
iterations = 100000

tf.compat.v1.reset_default_graph()
input_data1 = tf.compat.v1.placeholder(tf.int32, [batch_size, maxSeqLength])
input_data2 = tf.compat.v1.placeholder(tf.int32, [batch_size, maxSeqLength])
dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
#词向量层
with tf.variable_scope('embedding'):
	if w2v:
		embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
										 initializer=tf.constant_initializer(W_embedding), trainable=True)
	else:
		embedding = tf.Variable(
					tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
					trainable=True,name="W")

def BiRNN(x, dropout, scope, hidden_units):
	n_hidden=hidden_units
	n_layers=3
	x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))

	with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope,reuse=tf.AUTO_REUSE):
		stacked_rnn_fw = []
		for _ in range(n_layers):
			fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
			lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
			stacked_rnn_fw.append(lstm_fw_cell)
		lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)


	with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope,reuse=tf.AUTO_REUSE):
		stacked_rnn_bw = []
		for _ in range(n_layers):
			bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
			lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=dropout)
			stacked_rnn_bw.append(lstm_bw_cell)
		lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
	
	# Get lstm cell output
	with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope,reuse=tf.AUTO_REUSE):
		outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)

	return outputs[-1]

embedded_chars1 = tf.nn.embedding_lookup(embedding, input_data1)
embedded_chars2 = tf.nn.embedding_lookup(embedding, input_data2)

with tf.variable_scope('cnn_text'):
	out1 = BiRNN(embedded_chars1, 0.5, "side", hidden_units)
	out2 = BiRNN(embedded_chars2, 0.5, "side", hidden_units)

	distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(out1,out2)),1,keepdims=True))
	distance = tf.math.divide(distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(out1),1,keepdims=True)),tf.sqrt(tf.reduce_sum(tf.square(out2),1,keepdims=True))))
	distance = tf.reshape(distance, [-1], name="distance")

def contrastive_loss(y,d,batch_size):
	#当AB语义相似的二者欧式距离尽量要小
	#不相同二者的距离尽量要大
	tmp= y *tf.square(d)
	tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
	return tf.reduce_sum(tmp +tmp2)/batch_size/2

input_y = tf.compat.v1.placeholder(tf.float32, [batch_size], name='input_y')#将标题和摘要直接拼接
#损失函数
with tf.name_scope("loss"):
	#语义相似度,对比损失函数
	loss = contrastive_loss(input_y,distance, batch_size)
#优化器
optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)
with tf.name_scope("accuracy"):
    temp_sim = tf.subtract(tf.ones_like(distance),tf.rint(distance), name="temp_sim") #auto threshold 0.5
    correct_predictions = tf.equal(temp_sim, input_y)
    accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


import datetime
from tqdm import tqdm
sess = tf.compat.v1.InteractiveSession()
saver = tf.compat.v1.train.Saver()

saver.restore(sess, tf.train.latest_checkpoint('models'))
#sess.run(tf.compat.v1.global_variables_initializer())
for i in tqdm(range(iterations)):
	X1,X2,Y2 = data_handler.create_train_data(data_handler.pos_train_x,
									data_handler.neg_train_x,
									num=int(batch_size/2))
	sess.run(optimizer, {input_data1:X1,input_data2:X2,input_y:Y2})
	#Save the network every 10,000 training iterations
	if (i % 100 == 0):
		save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
		print("saved to %s" % save_path)
		X1,X2,Y2 = data_handler.create_train_data(data_handler.pos_test_x,
										data_handler.neg_test_x,
										num=int(batch_size/2))
		_accuracy = sess.run(accuracy, {input_data1:X1,input_data2:X2,input_y:Y2})
		print(_accuracy)
sess.close()