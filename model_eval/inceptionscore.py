from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys

img_path = "../../datasets/color/"

# img_path = "../../test-data/"

pretrained_model = './inception_retrained/inception_retrained_graph.pb'
softmax = None

def get_inception_score(images, splits=10):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  
  inps = []

  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  
  batch_size = 100 
  with tf.Session() as sess:
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(batch_size)))
    
    # input_shape = [None, None, None, 3]
    # x = tf.placeholder( tf.float32, input_shape, name='x')

    for i in range(n_batches):  # 570
        sys.stdout.write(".")
        sys.stdout.flush()
        
        inp = inps[(i*batch_size) : min((i+1)*batch_size, len(inps))]
        inp = np.concatenate(inp, 0)
        
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        # print(sess.graph.get_tensor_by_name('ExpandDims:0'))
       
        # pred = sess.run(softmax, {x: inp})
        # print(sess.graph.get_tensor_by_name('x:0'))
        
        preds.append(pred)
    
    preds = np.concatenate(preds, 0)

    file = open("inceptionScore_4real_data.txt", "a") #inceptionScore_4generated.txt if image_path was to generated datas
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
     
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))

      # kl = np.mean(part, axis=0)
      # kl = np.mean([entropy(part[i, :], kl) for i in range(part.shape[0])])
      
      scores.append(np.exp(kl))
      
      file.write( "\n" + str(np.mean(scores)) +" "+ str(np.std(scores)))
      print(np.mean(scores), np.std(scores))
    
    file.close()
    return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception():
  global softmax
  global input_shape
  global x
  
  with tf.gfile.FastGFile(pretrained_model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


  # weight_shape = [None, 18]
  # w = tf.placeholder( tf.float32, weight_shape, name='w')

  with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o._shape = tf.TensorShape(new_shape)
    
    # for op_idx, op in enumerate(ops):
    #     for o in op.outputs:
    #         shape = o.get_shape()
    #         if shape._dims != []:
    #           shape = [s.value for s in shape]
    #           new_shape = []
    #           for j, s in enumerate(shape):
    #             if s == 1 and j == 0:
    #               new_shape.append(None)
    #             else:
    #               new_shape.append(s)
    #           o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    
    w = sess.graph.get_operation_by_name("softmax_1/Wx_plus_b/MatMul").inputs[1]
    
    print('w', w.shape)
    print('pool3', tf.squeeze(pool3,  [1, 2]).shape)

    logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
    
    softmax = tf.nn.softmax(logits)

if __name__=='__main__':
    if softmax is None:
      _init_inception()

    def get_images(image_path):
      image_array = []
      for path in os.listdir(image_path):
        if not path.startswith('.'):
            for filename in os.listdir(image_path+path):
              if not filename.startswith('.'):
                image_array.append(scipy.misc.imread(image_path+path +'/'+filename))
      return image_array
    
    print("\n# images : ",str(len(get_images(img_path))), " IS (mean stddv) : ",
         get_inception_score(get_images(img_path)))
