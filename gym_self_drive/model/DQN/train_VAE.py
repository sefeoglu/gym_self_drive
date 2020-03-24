import tensorflow as tf
import numpy as np
import glob, random, os
from model import VariationalAutoencoderConfig1
from model import VariationalAutoencoderConfig2
from model import VariationalAutoencoderConfig3
from model import VariationalAutoencoderConfig4
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = "./gym_self_drive/model/saved_models/"
MODEL_NAME = MODEL_PATH + 'model'

class TrainModel(object):
    ''' Train the your own model'''
    def __init__(self, model):
        self.network = model
        pass

    def data_iterator(self, batch_size):
        data_files = glob.glob('./gym_self_drive/model/data/obs_data_VAE_*')
        while True:
            data = np.load(random.sample(data_files, 1)[0])
            np.random.shuffle(data)
            np.random.shuffle(data)
            N = data.shape[0]
            start = np.random.randint(0, N-batch_size)
            yield data[start:start+batch_size]

    def train_vae(self):
        sess = tf.InteractiveSession()

        global_step = tf.Variable(0, name='global_step', trainable=False)

        writer = tf.summary.FileWriter('logdir')

        
        train_op = tf.train.AdamOptimizer(0.001).minimize(self.network.loss, global_step=global_step)
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(max_to_keep=1)
        step = global_step.eval()
        training_data = self.data_iterator(batch_size=128)

        try:
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
            print("Model restored from: {}".format(MODEL_PATH))
        except:
            print("Could not restore saved model")

        try:
            while True:
                images = next(training_data)
                _, loss_value, summary = sess.run([train_op, self.network.loss, self.network.merged],
                                    feed_dict={self.network.image: images})
                writer.add_summary(summary, step)

                if np.isnan(loss_value):
                    raise ValueError('Loss value is NaN')
                if step % 10 == 0 and step > 0:
                    print ('step {}: training loss {:.6f}'.format(step, loss_value))
                    save_path = saver.save(sess, MODEL_NAME, global_step=global_step)
                if loss_value <= 35:
                    print ('step {}: training loss {:.6f}'.format(step, loss_value))
                    save_path = saver.save(sess, MODEL_NAME, global_step=global_step)
                    break
                step+=1

        except (KeyboardInterrupt, SystemExit):
            print("Manual Interrupt")

        except Exception as e:
            print("Exception: {}".format(e))


    def load_vae(self):

        graph = tf.Graph()
        with graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config, graph=graph)

            init = tf.global_variables_initializer()
            sess.run(init)

            saver = tf.train.Saver(max_to_keep=1)
            training_data = self.data_iterator(batch_size=128)

            try:
                saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
            except:
                raise ImportError("Could not restore saved model")

            return sess, self.network

if __name__ == "__main__":
    # add model selection step here
    NETWOK = VariationalAutoencoderConfig2()
    train = TrainModel(NETWOK)
    train.train_vae()
    
