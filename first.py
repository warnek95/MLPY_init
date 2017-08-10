from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

########### matplotlibrc location
# import matplotlib
# print(matplotlib.matplotlib_fname())
###############

############# available matplotlib backend
# import matplotlib.rcsetup as rcsetup
# print(rcsetup.all_backends)
#############

iris = load_iris()
print(iris['DESCR'][:193] + "\n...")
print(iris['target_names'])

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

############## Plot to see data coherence
# fig, ax = plt.subplots(3, 3, figsize=(15, 15))
# plt.suptitle("iris_pairplot")
#
# for i in range(3):
#     for j in range(3):
#         ax[i, j].scatter(X_train[:, j], X_train[:, i + 1], c=y_train, s=60)
#         ax[i, j].set_xticks(())
#         ax[i, j].set_yticks(())
#         if i == 2:
#             ax[i, j].set_xlabel(iris['feature_names'][j])
#         if j == 0:
#             ax[i, j].set_ylabel(iris['feature_names'][i + 1])
#         if j > i:
#             ax[i, j].set_visible(False)
# plt.show()
