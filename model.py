from ops import *
class DenseNet():
    def __init__(self, nb_blocks, filters, dropout_rate,training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.dropout_rate=dropout_rate
        self.training = training



    def bottleneck_layer(self, x, scope):
        with tf.name_scope(scope):
            x = batch_normalization(x, training=self.training, scope=scope+'_batch1')
            x = relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[3,3], layer_name=scope+'_conv1')
            x = drop_out(x, rate=self.dropout_rate, training=self.training)

            x = batch_normalization(x, training=self.training, scope=scope+'_batch2')
            x = relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = drop_out(x, rate=self.dropout_rate, training=self.training)
            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = batch_normalization(x, training=self.training, scope=scope+'_batch1')
            x = relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = drop_out(x, rate=self.dropout_rate, training=self.training)
            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)
            x = concatenation(layers_concat)
            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[3,3], stride=1, layer_name='conv0')

        x = self.dense_block(input_x=x, nb_layers=5, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=5, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=5, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')

        x = self.dense_block(input_x=x, nb_layers=5, layer_name='dense_final')
        x = batch_normalization(x, training=self.training, scope='linear_batch')
        x = relu(x)
        x = conv_layer(x, filter=4 * self.filters, kernel=[3, 3], stride=1, layer_name='conv1')
        x = conv_layer(x, filter=1, kernel=[3, 3], stride=1, layer_name='conv2')

        return x



class single_gpu_model(object):
    def __init__(self,sess,gpu_list,config):
        self.sess=sess
        self.gpu_list=gpu_list
        self.is_train=tf.placeholder(tf.bool,name='training_flag')
        self.nb_blocks = config.nb_blocks
        self.filters = config.filters
        self.dropout_rate = config.dropout_rate
        self.learning_rate=tf.placeholder(tf.float32, shape=[])
        self.opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        self.weight_decay=config.weight_decay
        self.models = []
        self.densenet=DenseNet(nb_blocks=self.nb_blocks, filters=self.filters, dropout_rate=self.dropout_rate,training=self.is_train)

        self.x = tf.placeholder(tf.float32, (None, None, None, 1), name='x')
        self.y = tf.placeholder(tf.float32, (None, None, None, 1), name='y')

        self.pred = self.densenet.Dense_net(self.x)
        self.mse_loss = loss_cost(self.pred, self.y)
        self.psnr = PSNR_cal(self.pred, self.y)
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        self.total_loss = self.mse_loss+ l2_loss * self.weight_decay
        self.grads = self.opt.compute_gradients(self.total_loss)
        for i, (g, v) in enumerate(self.grads):
            if g is not None:
                self.grads[i] = (tf.clip_by_norm(g, 0.1), v)
        self.saver = tf.train.Saver()

    def fit(self,batch_x,batch_y,lr,train_phase):
        self.sess.run(tf.global_variables_initializer())
        inp_dict={}
        inp_dict[self.learning_rate]=lr
        inp_dict[self.is_train]=train_phase
        inp_dict[self.x]=batch_x
        inp_dict[self.y]=batch_y
        _, _loss, _psnr, _mse = self.sess.run([self.grads, self.total_loss, self.psnr, self.mse_loss], inp_dict)

        return _loss,_psnr,_mse

    def deploy(self,x,y,train_phase):
        inp_dict = {}
        inp_dict[self.is_train] = train_phase
        inp_dict[self.x] = x
        inp_dict[self.y] = y
        test_total_loss, test_psnr_loss, test_mse_loss = self.sess.run([self.total_loss, self.psnr, self.mse_loss], inp_dict)
        return test_total_loss, test_psnr_loss, test_mse_loss
    def save(self,ckpt_path):
        self.saver.save(self.sess,ckpt_path+'/model.ckpt')
    def restore(self,ckpt_path):
        self.saver.restore(self.sess,ckpt_path+'/model.ckpt')




























