


- This project mainy focues on Trasfer leraning ;
- In transfer learning, we first train a base network on a base dataset and task, and then we repurpose the learned features, or transfer them, to a second target network to be trained on a target dataset and task. This process will tend to work if the features are general, meaning suitable to both base and target tasks, instead of specific to the base task.;
![](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/09/Depiction-of-Inductive-Transfer.png)
- In this porject i used  VGG16
VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”
![](https://neurohive.io/wp-content/uploads/2018/11/vgg16.png)

![](https://neurohive.io/wp-content/uploads/2018/11/vgg16-neural-network.jpg)

- The input to cov1 layer is of fixed size 224 x 224 RGB image. The image is passed through a stack of convolutional (conv.) layers, where the filters were used with a very small receptive field: 3×3 (which is the smallest size to capture the notion of left/right, up/down, center). In one of the configurations, it also utilizes 1×1 convolution filters, which can be seen as a linear transformation of the input channels (followed by non-linearity). The convolution stride is fixed to 1 pixel; the spatial padding of conv. layer input is such that the spatial resolution is preserved after convolution, i.e. the padding is 1-pixel for 3×3 conv. layers. Spatial pooling is carried out by five max-pooling layers, which follow some of the conv.  layers (not all the conv. layers are followed by max-pooling). Max-pooling is performed over a 2×2 pixel window, with stride 2.


### Requirements
- python
- Tensorflow ( cpu , gpu)
- opencv
- pretrained moedl (vgg_face_weights.h5) - 550MB
- cascade classifier( haarcascade_frontalface_default.xml)
- images with name in test_samples folder


### Testing

- Where testing can be done in both image to image or video to image
- Loss function used are Cosine_similarity and Equlidean-distance both are 
placed in loss file

![](https://i.ibb.co/JKRhw2R/snipping1.png)
![](https://i.ibb.co/Ny3scWs/snipping2.png)
- VGG trained aroud 141 million instances

![](https://i1.wp.com/sefiks.com/wp-content/uploads/2018/08/angelina-jolie-true-positive-v2.png?ssl=1)


### Loss Fuction
    
```python
def cosine(X, Y):
    a = np.matmul(np.transpose(X), Y)
    b = np.sum(np.multiply(X, X))
    c = np.sum(np.multiply(Y, Y))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
```
	

- similarity set to 0.4 between both iamges to get better for identifiaction




