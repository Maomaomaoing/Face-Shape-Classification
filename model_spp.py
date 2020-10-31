import tensorflow as tf
import math

class MODEL:
    def __init__(self, LR, FILTER_NUM, BATCH_SIZE, TEST_BATCH_SIZE, REUSE, STATE, N_CLASSES):
        self.LR = LR
        self.REUSE = REUSE
        self.STATE = STATE
        self.BATCH_SIZE = BATCH_SIZE
        self.TEST_BATCH_SIZE = TEST_BATCH_SIZE
        self.FILTER_NUM = FILTER_NUM
        self.N_CLASSES = N_CLASSES
        self.TRAINING_FLAG = True
        
        with tf.variable_scope('inputs'):
            self.img = tf.placeholder(tf.string, [None], name = 'img')
            self.label = tf.placeholder(tf.int32, [None], name = 'label')
            
            self.input_img = tf.placeholder(tf.float32, [None, 320, 320, 3], name = 'input_img')
            self.input_label = tf.placeholder(tf.int32, [None], name = 'input_label')
        
        def img_preprocess(x):
            x = tf.cast( tf.image.decode_jpeg( tf.read_file(x) , try_recover_truncated = True)[:,:,:3], tf.float32)
            return tf.reshape( tf.image.resize(x, (320, 320), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False), (320, 320, 3))
        def normalize(x):
            return (x - 127.5)/127.5
            
        
        #self.img_list = tf.map_fn(img_preprocess, self.input, dtype=tf.float32, name = 'map_function')

        self.dataset = tf.data.Dataset.from_tensor_slices({'imgs': self.img, 'labs': self.label})
        self.dataset = self.dataset.map(lambda x: {'imgs': img_preprocess(x['imgs']), 'labs': x['labs']})
        self.dataset = self.dataset.batch(self.BATCH_SIZE, drop_remainder = True)
        self.dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat ( buffer_size=self.BATCH_SIZE ) )
        self.dataset = self.dataset.prefetch( buffer_size=self.BATCH_SIZE )
        self.dataset_iter = self.dataset.make_initializable_iterator()
        self.dataset_fetch = self.dataset_iter.get_next()
        
        self.dataset_test = tf.data.Dataset.from_tensor_slices({'imgs': self.img, 'labs': self.label})
        self.dataset_test = self.dataset_test.map(lambda x: {'imgs': img_preprocess(x['imgs']), 'labs': x['labs']})
        # test batch size = 26
        self.dataset_test = self.dataset_test.batch(self.TEST_BATCH_SIZE, drop_remainder = True)
        self.dataset_test = self.dataset_test.apply(tf.contrib.data.shuffle_and_repeat ( buffer_size=self.TEST_BATCH_SIZE ) )
        self.dataset_test = self.dataset_test.prefetch( buffer_size=self.TEST_BATCH_SIZE )
        self.dataset_iter_test = self.dataset_test.make_initializable_iterator()
        self.dataset_fetch_test = self.dataset_iter_test.get_next()
        
        
        with tf.variable_scope('model'):
            self.output = self.main(normalize(self.input_img))
            self.output = tf.identity(self.output, 'output')
            
        with tf.variable_scope('result'):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.input_label, logits = self.output), name = 'loss')
            self.prediction = tf.cast(tf.argmax(self.output, axis=1), tf.int32, name = 'prediction')
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.input_label, dtype=tf.int32), self.prediction), tf.float32), name = 'accuracy')
            
        self.train_op = tf.train.AdamOptimizer(self.LR).minimize(self.loss)
        

    def main(self, img):
        
        with tf.variable_scope('dilated_conv', reuse = tf.AUTO_REUSE) :
            
            p = "same"
            # act = tf.nn.selu
            act = tf.nn.tanh
            # k_init = tf.keras.initializers.lecun_normal()
            k_init = lambda i, o: tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=math.sqrt(2.0 / (i+o)))
            flag = self.TRAINING_FLAG

            conv2d = lambda num_f, k_size, pad, act, k_init, strd: \
            tf.keras.layers.Conv2D(num_f, k_size, padding=pad, activation=act,  kernel_initializer=k_init, strides=strd)
            batch_norm = lambda: tf.keras.layers.BatchNormalization()

            print("ori image", img.shape)

            conv1 = conv2d(96,7,p,act,k_init(3,96),2)(img)
            pool1 = tf.keras.layers.MaxPool2D(3, padding=p, strides=2)(conv1)
            print("block1", pool1.shape)

            conv2 = conv2d(256,5,p,act,k_init(96,256),2)(pool1)
            pool2 = tf.keras.layers.MaxPool2D(3, padding=p, strides=2)(conv2)
            print("block2", pool2.shape)
     
            conv3 = conv2d(384,3,p,act,k_init(256,384),1)(pool2)
            conv4 = conv2d(384,3,p,act,k_init(384,384),1)(conv3)
            conv5 = conv2d(256,3,p,act,k_init(384,256),1)(conv4)
            print("block3", conv5.shape)

            ## shape: 20*20 -> 4*4, 2*2, 1*1
            spp16 = tf.keras.layers.MaxPool2D(5, padding="valid")(conv5)
            spp4 = tf.keras.layers.MaxPool2D(10, padding="valid")(conv5)
            spp1 = tf.keras.layers.MaxPool2D(20, padding="valid")(conv5)
            print("spp shape", spp16.shape, spp4.shape, spp1.shape)

            flat1 = tf.keras.layers.Flatten()(spp16)
            flat2 = tf.keras.layers.Flatten()(spp4)
            flat3 = tf.keras.layers.Flatten()(spp1)

            flat = tf.concat([flat1, flat2, flat3], axis=-1)
            print("flat shape", flat.shape)

            # fc1 = tf.keras.layers.Dense(1024, kernel_initializer = k_init(1,1), activation = act)(flat)

            fc2 = tf.keras.layers.Dense(1024, kernel_initializer = k_init(1,1), activation = act)(flat)
            fc3 = tf.keras.layers.Dense(5, kernel_initializer = k_init(1,1))(fc2)
            return fc3
        
        