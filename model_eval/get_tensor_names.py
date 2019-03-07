import tensorflow as tf

def printTensors(pb_file):

    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # # print operations
    for op in graph.get_operations():
        print(op.values()) # op.name for tensor names
    
if __name__ == "__main__":

    # printTensors("./imagenet/classify_image_graph_def.pb")
    # printTensors("pretrained_models/Inception_Scratch_95.pb")
    printTensors("Image-classification-transfer-learning/inception_retrained_graph.pb")
