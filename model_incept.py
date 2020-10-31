import tensorflow as tf

class MODEL:
    def __init__(self, LR, FILTER_NUM, BATCH_SIZE, TEST_BATCH_SIZE, REUSE, STATE, N_CLASSES):
        self.LR = LR
        self.REUSE = REUSE
        self.STATE = STATE
        self.BATCH_SIZE = BATCH_SIZE
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
        self.dataset_test = self.dataset_test.batch(1, drop_remainder = True)
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
            cfg_3a = [64, 96, 128, 16, 32, 32]
            cfg_3b = [128, 128, 196, 32, 96, 64]
            cfg_4a = [192, 96, 208, 16, 48, 64]
            cfg_4b = [160, 112, 224, 24, 64, 64]
            cfg_4c = [128, 128, 256, 24, 64, 64]
            cfg_4d = [112, 144, 288, 32, 64, 64]
            cfg_4e = [256, 160, 320, 32, 128, 128]
            cfg_5a = [256, 160, 320, 32, 128, 128]
            cfg_5b = [384, 192, 384, 48, 128, 128]
            
            ## layers
            conv2d = lambda num_f, k_size, pad, act, k_init, stride: \
            tf.keras.layers.Conv2D(num_f, k_size, padding=pad, activation=act, kernel_initializer=k_init, strides=stride)
            def incept_layer(input, k_size):
                inc1 = conv2d(k_size,1,p,act,k_init,1)(input)
                inc2 = conv2d(k_size,1,p,act,k_init,1)(input)
                inc2 = conv2d(k_size,3,p,act,k_init,1)(inc2)
                inc3 = conv2d(k_size,1,p,act,k_init,1)(input)
                inc3 = conv2d(k_size,5,p,act,k_init,1)(inc3)
                inc4 = tf.keras.layers.MaxPool2D(3,1, padding=p)(input)
                inc4 = conv2d(k_size,1,p,act,k_init,1)(inc4)
                concat = tf.concat([inc1, inc2, inc3, inc4], axis=-1)
                print("concated shape", concat.shape)
                return concat
            
            conv1 = conv2d(64,7,p,act,k_init,2)(img)
            pool1 = tf.keras.layers.MaxPooling2D(3, 2, padding=p)(conv1)
            print("pool1 shape", pool1.shape)

            conv2 = conv2d(64,1,p,act,k_init,1)(pool1)
            conv3 = conv2d(192,3,p,act,k_init,1)(conv2)
            pool2 = tf.keras.layers.MaxPooling2D(3,2, padding=p)(conv3)
            print("pool2 shape", pool2.shape)

            incpt1 = incept_layer(pool2, cfg_3a)
            incpt2 = incept_layer(incpt1, cfg_3b)
            pool3 = tf.keras.layers.MaxPooling2D(3,2, padding=p)(incpt2)
            print("pool3 shape", pool3.shape)

            incpt3 = incept_layer(pool3, cfg_4a)
            incpt4 = incept_layer(incpt3, cfg_4b)
            incpt5 = incept_layer(incpt4, cfg_4c)
            incpt6 = incept_layer(incpt5, cfg_4d)
            incpt7 = incept_layer(incpt6, cfg_4e)
            pool4 = tf.keras.layers.MaxPooling2D(3,2, padding=p)(incpt7)
            print("pool4 shape", pool4.shape)

            incpt8 = incept_layer(pool4, cfg_5a)
            incpt9 = incept_layer(incpt8, cfg_5b)
            pool5 = tf.keras.layers.AveragePooling2D(7,1, padding="valid")(incpt9)
            print("pool5 shape", pool5.shape)

            flat = tf.keras.layers.Flatten()(pool5)
            print("flat shape", flat.shape)

            fc1 = tf.keras.layers.Dense(4096, kernel_initializer = k_init, activation = act)(flat)
            fc2 = tf.keras.layers.Dense(5)(fc1)
            return fc2
        
        