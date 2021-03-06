import tensorflow as tf

class MODEL:
    def __init__(self, LR, FILTER_NUM, BATCH_SIZE, TEST_BATCH_SIZE, REUSE, STATE, N_CLASSES):
        self.LR = LR
        self.REUSE = REUSE
        self.STATE = STATE
        self.BATCH_SIZE = BATCH_SIZE
        self.TEST_BATCH_SIZE = TEST_BATCH_SIZE
        self.FILTER_NUM = FILTER_NUM
        self.N_CLASSES = N_CLASSES
        
        self.act = tf.nn.selu
        self.kernel = tf.keras.initializers.lecun_normal()
        
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
        
        with tf.variable_scope('vgg16') :

            p = "same"
            act = tf.nn.selu
            k_init = tf.keras.initializers.lecun_normal()
            
            conv2d = lambda num_f, k_size, pad, act, k_init: \
            tf.keras.layers.Conv2D(num_f, k_size, padding=pad, activation=act,  kernel_initializer=k_init)
            batch_norm = lambda: tf.keras.layers.BatchNormalization()

            conv1 = conv2d(64,3,p,act,k_init)(img)
            conv2 = conv2d(64,3,p,act,k_init)(conv1)
            conv2 = batch_norm()(conv2)
            pool1 = tf.keras.layers.MaxPool2D(2, padding=p)(conv2)

            conv3 = conv2d(128,3,p,act,k_init)(pool1)
            conv4 = conv2d(128,3,p,act,k_init)(conv3)
            conv4 = batch_norm()(conv4)
            pool2 = tf.keras.layers.MaxPool2D(2, padding=p)(conv4)

            conv5 = conv2d(256,3,p,act,k_init)(pool2)
            conv6 = conv2d(256,3,p,act,k_init)(conv5)
            conv7 = conv2d(256,3,p,act,k_init)(conv6)
            conv8 = conv2d(256,3,p,act,k_init)(conv7)
            conv8 = batch_norm()(conv8)
            pool3 = tf.keras.layers.MaxPool2D(2, padding=p)(conv8)

            conv9 = conv2d(512,3,p,act,k_init)(pool3)
            conv10 = conv2d(512,3,p,act,k_init)(conv9)
            conv11 = conv2d(512,3,p,act,k_init)(conv10)
            conv12 = conv2d(512,3,p,act,k_init)(conv11)
            conv12 = batch_norm()(conv12)
            pool4 = tf.keras.layers.MaxPool2D(2, padding=p)(conv12)
            
            conv13 = conv2d(512,3,p,act,k_init)(pool4)
            conv14 = conv2d(512,3,p,act,k_init)(conv13)
            conv15 = conv2d(512,3,p,act,k_init)(conv14)
            conv16 = conv2d(512,3,p,act,k_init)(conv15)
            conv16 = batch_norm()(conv16)
            pool5 = tf.keras.layers.MaxPool2D(2, padding=p)(conv16)
            
            flat = tf.keras.layers.Flatten()(pool5)
            print("flat shape", flat.shape)

            fc1 = tf.keras.layers.Dense(4096, kernel_initializer = k_init, activation = act)(flat)
            fc2 = tf.keras.layers.Dense(512, kernel_initializer = k_init, activation = act)(fc1)
            fc3 = tf.keras.layers.Dense(5)(fc2)
            return fc3
        
        