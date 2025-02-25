import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

(mnistf_train_x,mnistf_train_y),(mnistf_test_x,mnistf_test_y) = fashion_mnist.load_data()
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
for i in range(10):
    plt.subplot(2, 5, i + 1)
    img = mnistf_train_x[mnistf_train_y==i][0]
    plt.imshow(img,cmap='gray')
    plt.title(labels[i])
    plt.axis('off')
plt.tight_layout()
plt.show()