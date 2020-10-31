import os
# os.chdir("/home/0756728/code/face_sshape/")
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from model_spp import MODEL

dataset_path = "data/face/"

#hyperparameter
EPOCH = 3000
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1
FILTER_NUM = 32
LR = 2e-5

path = os.path.join(os.getcwd(), dataset_path)
class_name = [os.path.join(path, i) for i in os.listdir(path)]
file = [[os.path.join(path, c_n, c) for c in os.listdir(c_n)] for c_n in class_name]
label = [[f]*len(file[f]) for f in range(len(file))]
print("The labels are: ", len(label))

#make training data
def data():
	training = []; testing = []
	training_label = []; testing_label = []
	for i in range(len(file)):
		np.random.shuffle(file[i]); np.random.shuffle(label[i])
		training.append( file[i][:int(len(file[i])*9/10)] )
		testing.append( file[i][int(len(file[i])*9/10):] )
		training_label.append( label[i][:int(len(label[i])*9/10)] )
		testing_label.append( label[i][int(len(label[i])*9/10):])
	return training, testing, training_label, testing_label

def upsampling(data, label):
	n_max = np.max([len(c) for c in data])
	new_data = []; new_label = []
	for i in range(len(data)):
		dul_data = data[i] * int(n_max / len(data[i]))
		dul_label = label[i] * int(n_max / len(label[i]))
		print("origin data:", len(data[i]), len(label[i]), "new data:", len(dul_data), len(dul_label))
		new_data.extend(dul_data)
		new_label.extend(dul_label)
	return new_data, new_label


training, all_testing, training_label, all_testing_label = data()
training, training_label = upsampling(training, training_label)

print("training number: ", len(training))
print("training label number: ", len(training_label))
n_test = sum([len(c) for c in all_testing])
print("testing number: ", n_test)
print("testing label number: ", [len(c) for c in all_testing_label], sum([len(c) for c in all_testing_label]))


os.environ['TF_CPP_MIN_LOG_LEVEL']='1' #do not show run error
os.environ['CUDA_VISIBLE_DEVICES']='5' #use which GPU device
warnings.filterwarnings('ignore')
print('This is EPOCH :', EPOCH)

# create session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

tf.reset_default_graph()
model = MODEL(LR, FILTER_NUM, BATCH_SIZE, TEST_BATCH_SIZE, True, 'training', len(label))
saver = tf.train.Saver(max_to_keep = 20)

try: os.mkdir("ckpt_model")
except: pass

with tf.Session(config=config) as sess:   
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	
	#training 
	print('Total steps: ' + str( int( len(training) / BATCH_SIZE ) ) )
	
	sess.run(model.dataset_iter.initializer, feed_dict={model.img: training, model.label: training_label})
	# sess.run(model.dataset_iter_test.initializer, feed_dict={model.img: testing, model.label: testing_label})
	output_res = []
	for e in range(EPOCH):
		total_loss = []
		train_acc = []
		for step in range( int( len(training) / BATCH_SIZE ) ):
			data_dict = sess.run(model.dataset_fetch)
			batch_data = data_dict['imgs']; batch_label = data_dict['labs']
			_, loss, acc, pred = sess.run([model.train_op, model.loss, model.accuracy, model.output], feed_dict={model.input_img: batch_data, model.input_label: batch_label})
			total_loss.append(loss)
			train_acc.append(acc)
		
		print('epoch: ', e, ' loss: ', np.mean(total_loss), ' acc: ', np.mean(train_acc))
			
		if (e+1) % 5 ==0 or e == 0: 
			test_acc = []
			for testing, testing_label in zip(all_testing, all_testing_label):
				## feed one class data
				sess.run(model.dataset_iter_test.initializer, feed_dict={model.img: testing, model.label: testing_label})
				acc_write= []
				for step in range( int( len(testing) / TEST_BATCH_SIZE ) ):
					data_dict = sess.run(model.dataset_fetch_test)
					batch_data = data_dict['imgs']; batch_label = data_dict['labs']
					loss, acc, pred = sess.run([model.loss, model.accuracy, model.output], feed_dict={model.input_img: batch_data, model.input_label: batch_label})
					acc_write.append(acc)
				test_acc.append(np.mean(acc_write))
			if np.mean(train_acc) > 0.95:
				saver.save(sess, './spp_model/'+ 'face_' + str(e+1) + '.ckpt')
			print("5 class test accuracy: ", test_acc, "total accuracy:", np.mean(test_acc))
			output_res.append([e, np.mean(total_loss), np.mean(train_acc), np.mean(test_acc)] + test_acc)
			out_df = pd.DataFrame(columns = ["epoch", "loss", "train acc", "test acc"] + class_name, data = output_res)
			out_df.to_csv("res_spp.csv", index=False)


		
		