#import tensorflow as tf
#import numpy as np
from glob import glob
import os
import random
import time


batch_size = 64
img_height = 64
img_width = 64

#train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#            "/home/cv307/U2020/BEGAN-CS/data/CelebA/train",seed=123,
#            image_size=(img_height, img_width), batch_size=batch_size)

#class_names = train_ds.class_names

data = glob(os.path.join("/home/cv307/U2020/BEGAN-CS/data/CelebA/train/*/*.jpg"))
class_num = len(os.listdir("/home/cv307/U2020/BEGAN-CS/data/CelebA/train/"))

start = time.time()

label = []
cnt = dict.fromkeys(range(class_num + 2), 0)
for i in range(0, len(data)):
    d, _ = os.path.split(data[i])
    _, l = os.path.split(d)
    label.append(int(l))
    cnt[label[i]] += 1
    #print(data[i] + ", " + str(label[i]))

#print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in cnt.items()) + "}")
#anchor = []

for i in range(0, len(data)):
    a = random.randint(0, cnt[label[i]])
    f = -1
    while(1):
        n = random.randint(0, len(data))
        if label[n] != label[i]:
            negative = data[n]
            #_, negative = os.path.split(data[n])
            break
    for j in range(0, len(data)):
        if label[j] == label[i]:
            if f == a:
                anchor = data[j]
                #_, anchor = os.path.split(data[j])
                #print(str(label[i]) + ", " + anchor + ", " + negative)
                break
            else:
                f += 1

#anchor.clear()
print(time.time() - start)

'''
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    print(image_batch[0])
    print(labels_batch[0])
    break

label = np.concatenate([i for x, i in train_ds])

for x, i in train_ds:
    (h, w, _) = x[0].shape
    break
print(h)
'''
