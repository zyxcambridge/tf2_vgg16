#coding=utf-8
import  os
import  tensorflow as tf
from    tensorflow import  keras
from    tensorflow.keras import datasets, layers, optimizers
import  argparse
import  numpy as np



from    network import VGG16
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
parser = argparse.ArgumentParser()

parser.add_argument('--imagedir', '-i',type=str, default='test_pic',
                           help="image_path_dir.")
# parser.add_argument('--image',type=str, default='test_pic',
#                            help="image_path_dir.")

cifar10_labels = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


def prepare_cifar(x, y):

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    return x, y


def main():

    args = parser.parse_args()

    tf.random.set_seed(22)
    print('loading data...')
    (x,y), (x_test_ori, y_test) = datasets.cifar10.load_data()
    X_train = x / 255.
    X_test_normal = x_test_ori / 255.
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    print('mean:', mean, 'std:', std)


    model = VGG16([32, 32, 3])

    model.load_weights('easy_checkpoint')

    test_imge_dir = args.imagedir
    print(test_imge_dir)

    for root, dirs, files in os.walk(test_imge_dir, topdown=False):
        for name in files:
            file_name = os.path.join(root, name)
            print(file_name)
            test_imge = cv2.imread(file_name)
            test_imge = test_imge / 255.
            test_imge_one = (test_imge - mean) / (std + 1e-7)
            input_x = np.expand_dims(test_imge_one,0)
            logits = model.predict(input_x)
            result_index = np.argmax(logits)
            print('predict value: ' + str(cifar10_labels[result_index]) + '\n')

    exit(0)

    img_index_all = 0
    img_index_err = 0
    x_test_dataset = (X_test_normal - mean) / (std + 1e-7)
    test_loader = tf.data.Dataset.from_tensor_slices((x_test_dataset, y_test))
    test_loader = test_loader.map(prepare_cifar).shuffle(10000).batch(1)
    for x_test_input, y in test_loader:
        # [b, 1] => [b]
        y = tf.squeeze(y, axis=1)
        print('true label: ' + str(cifar10_labels[y[0]]))
        # # [b, 10]
        # y = tf.one_hot(y, depth=10)

        logits = model.predict(x_test_input)
        result_index = np.argmax(logits)
        print('predict label: ' + str(cifar10_labels[result_index]) + '\n')
        if result_index != y[0]:
            x_test_ori = (x_test_input*(std + 1e-7) + mean)*255
            cv2.imwrite('error_result/{}_true_label_{}_predict_{}.jpg'.format(img_index_err,cifar10_labels[y[0]],
                                                                           cifar10_labels[result_index]),x_test_ori[0].numpy())
            img_index_err = img_index_err + 1
        else:
            x_test_ori = (x_test_input*(std + 1e-7) + mean)*255
            cv2.imwrite('good_result/{}_true_label_{}_predict_{}.jpg'.format(img_index_err,cifar10_labels[y[0]],
                                                                           cifar10_labels[result_index]),x_test_ori[0].numpy())
        img_index_all = img_index_all + 1
    print('所有图片数量 : {} 预测错误数量: {}'.format(img_index_all,img_index_err))
    exit(0)

if __name__ == '__main__':
    main()