import os
import json
from glob import glob
import cmath
import numpy as np
import seaborn as sns
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile

# %matplotlib inline

with gfile.GFile('/home/sly/PycharmProjects/lcd/classify_image_graph_def.pb', 'rb') as f:
    # graph_def = tf.GraphDef()
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

        # print operations
    for op in graph.get_operations():
        print(op.name)

def forward_pass(fname, target_layer='pool_3/_reshape:0'):
    g = tf.Graph()

    image_data = tf.io.gfile.GFile(fname, 'rb').read()

    with tf.compat.v1.Session(graph=g) as sess:
        tf.import_graph_def(graph_def, name='')

        pool3 = sess.graph.get_tensor_by_name(target_layer)
        pool3 = sess.run(pool3,
                         {'DecodeJpeg/contents:0': image_data})
        print(pool3)
        # print(pool3.shape)
        # print(pool3.ndim)
        print(np.squeeze(pool3))

        return pool3.flatten()

from IPython.display import Image
Image(filename='ferrari.jpg', width=300)
ferrari_repr = forward_pass('ferrari.jpg')
# print(ferrari_repr)
print(ferrari_repr.shape)

# Use [::2] to keep images from the left camera only
# filenames = sorted(glob('/home/sly/PycharmProjects/lcd/Images/*.jpg'))[::2]
#
# representations = []
#
# for fname in filenames:
#     frame_repr = forward_pass(fname)
#     representations.append(frame_repr.flatten())
# #   print(representations)
# print(len(representations))
# print(representations)

# def normalize(x): return x / np.linalg.norm(x)
#
#
# def build_confusion_matrix():
#     n_frames = len(representations)
#
#     confusion_matrix = np.zeros((n_frames, n_frames))
#
#     for i in range(n_frames):
#         for j in range(n_frames):
#             # confusion_matrix[i][j] = 1.0 - np.sqrt(
#             #     1.0 - np.dot(normalize(representations[i]), normalize(representations[j])))
#             confusion_matrix[i][j] = np.linalg.norm(representations[i]-representations[j])
#     return confusion_matrix
#
#
# confusion_matrix = build_confusion_matrix()
#
# # Load the ground truth
#
# GROUND_TRUTH_PATH = os.path.expanduser(
#     '/home/sly/PycharmProjects/lcd/NewCollegeGroundTruth.mat')
#
# gt_data = sio.loadmat(GROUND_TRUTH_PATH)['truth'][::2, ::2]
#
# # Set up plotting
#
# default_heatmap_kwargs = dict(
#     xticklabels=False,
#     yticklabels=False,
#     square=True,
#     cbar=False,)
#
# fig, (ax1, ax2) = plt.subplots(ncols=2)
#
# # Plot ground truth
# sns.heatmap(gt_data,
#     ax=ax1,
#     **default_heatmap_kwargs)
# ax1.set_title('Ground truth')
#
#
# # Only look at the lower triangle
# confusion_matrix = np.tril(confusion_matrix, 0)
#
# sns.heatmap(confusion_matrix,
#            ax=ax2,
#            **default_heatmap_kwargs)
# ax2.set_title('CNN')
#
#
# #
# # prec_recall_curve = []
# #
# # for thresh in np.arange(0, 0.75, 0.02):
# #     # precision: fraction of retrieved instances that are relevant
# #     # recall: fraction of relevant instances that are retrieved
# #     true_positives = (confusion_matrix > thresh) & (gt_data == 1)
# #     all_positives = (confusion_matrix > thresh)
# #
# #     try:
# #         precision = float(np.sum(true_positives)) / np.sum(all_positives)
# #         recall = float(np.sum(true_positives)) / np.sum(gt_data == 1)
# #
# #         prec_recall_curve.append([thresh, precision, recall])
# #     except:
# #         break
# #
# # prec_recall_curve = np.array(prec_recall_curve)
# #
# # plt.plot(prec_recall_curve[:, 1], prec_recall_curve[:, 2])
# #
# # for thresh, prec, rec in prec_recall_curve[25::5]:
# #     plt.annotate(
# #         str(thresh),
# #         xy=(prec, rec),
# #         xytext=(8, 8),
# #         textcoords='offset points')
# #
# # plt.xlabel('Precision', fontsize=14)
# # plt.ylabel('Recall', fontsize=14)
#
# plt.show()