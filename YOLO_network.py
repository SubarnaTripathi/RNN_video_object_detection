import lasagne
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer

from lasagne.nonlinearities import LeakyRectify
    
from lasagne.layers import GlobalPoolLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
#from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayerDNN

from lasagne.layers import MaxPool2DLayer as PoolLayer
#from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear
from lasagne.utils import floatX

def build_network( ):
    
    net = {}
    net['input'] = InputLayer((None, 3, 448, 448))    
    net['conv1/7x7_s2'] = ConvLayer( net['input'],        64, 7, stride=2, pad=2, flip_filters=False, nonlinearity=LeakyRectify(0.1)) 

    net['pool1/2x2_s2'] = PoolLayer( net['conv1/7x7_s2'], pool_size=2, stride=2, ignore_border=False)   

    net['conv2/3x3_s1'] = ConvLayer( net['pool1/2x2_s2'], 192, 3, stride=1, pad=1, flip_filters=False, nonlinearity=LeakyRectify(0.1))   
    net['pool2/2x2_s2'] = PoolLayer( net['conv2/3x3_s1'], pool_size=2, stride=2, ignore_border=False)  

    net['conv3/1x1_s1'] = ConvLayer( net['pool2/2x2_s2'], 128, 1, stride=1, pad=0, flip_filters=False, nonlinearity=LeakyRectify(0.1))      
    net['conv4/3x3_s1'] = ConvLayer( net['conv3/1x1_s1'], 256, 3, stride=1, pad=1, flip_filters=False, nonlinearity=LeakyRectify(0.1))   
    net['conv5/1x1_s1'] = ConvLayer( net['conv4/3x3_s1'], 256, 1, stride=1, pad=0, flip_filters=False, nonlinearity=LeakyRectify(0.1))       
    net['conv6/3x3_s1'] = ConvLayer( net['conv5/1x1_s1'], 512, 3, stride=1, pad=1, flip_filters=False, nonlinearity=LeakyRectify(0.1))   

    net['pool3/2x2_s2'] = PoolLayer( net['conv6/3x3_s1'], pool_size=2, stride=2, ignore_border=False)  

    ## 4 - times
    net['conv7/1x1_s1'] = ConvLayer( net['pool3/2x2_s2'], 256, 1, stride=1, pad=0, flip_filters=False, nonlinearity=LeakyRectify(0.1))       
    net['conv8/3x3_s1'] = ConvLayer( net['conv7/1x1_s1'], 512, 3, stride=1, pad=1, flip_filters=False, nonlinearity=LeakyRectify(0.1))   

    net['conv9/1x1_s1'] = ConvLayer( net['conv8/3x3_s1'], 256, 1, stride=1, pad=0, flip_filters=False, nonlinearity=LeakyRectify(0.1))       
    net['conv10/3x3_s1'] = ConvLayer( net['conv9/1x1_s1'], 512, 3, stride=1, pad=1, flip_filters=False, nonlinearity=LeakyRectify(0.1))   

    net['conv11/1x1_s1'] = ConvLayer( net['conv10/3x3_s1'], 256, 1, stride=1, pad=0, flip_filters=False, nonlinearity=LeakyRectify(0.1))    
    net['conv12/3x3_s1'] = ConvLayer( net['conv11/1x1_s1'], 512, 3, stride=1, pad=1, flip_filters=False, nonlinearity=LeakyRectify(0.1))       

    net['conv13/1x1_s1'] = ConvLayer( net['conv12/3x3_s1'], 256, 1, stride=1, pad=0, flip_filters=False, nonlinearity=LeakyRectify(0.1))   
    net['conv14/3x3_s1'] = ConvLayer( net['conv13/1x1_s1'], 512, 3, stride=1, pad=1, flip_filters=False, nonlinearity=LeakyRectify(0.1))   
    ####  

    net['conv15/1x1_s1'] = ConvLayer( net['conv14/3x3_s1'], 512, 1, stride=1, pad=0, flip_filters=False, nonlinearity=LeakyRectify(0.1))        
    net['conv16/3x3_s1'] = ConvLayer( net['conv15/1x1_s1'], 1024, 3, stride=1, pad=1, flip_filters=False, nonlinearity=LeakyRectify(0.1))   

    ## maxpool  4 ===>
    net['pool4/2x2_s2'] = PoolLayer( net['conv16/3x3_s1'], pool_size=2, stride=2, ignore_border=False)  

    ## 2 - times
    net['conv17/1x1_s1'] = ConvLayer( net['pool4/2x2_s2'], 512, 1, stride=1, pad=0, flip_filters=False, nonlinearity=LeakyRectify(0.1))       
    net['conv18/3x3_s1'] = ConvLayer( net['conv17/1x1_s1'], 1024, 3, stride=1, pad=1, flip_filters=False, nonlinearity=LeakyRectify(0.1))   

    net['conv19/1x1_s1'] = ConvLayer( net['conv18/3x3_s1'], 512, 1, stride=1, pad=0, flip_filters=False, nonlinearity=LeakyRectify(0.1))      
    net['conv20/3x3_s1'] = ConvLayer( net['conv19/1x1_s1'], 1024, 3, stride=1, pad=1, flip_filters=False, nonlinearity=LeakyRectify(0.1))  
    ####

    net['conv21/3x3_s1'] = ConvLayer( net['conv20/3x3_s1'], 1024, 3, stride=1, pad=1, flip_filters=False, nonlinearity=LeakyRectify(0.1))   
    net['conv22/3x3_s2'] = ConvLayer( net['conv21/3x3_s1'], 1024, 3, stride=2, pad=1, flip_filters=False, nonlinearity=LeakyRectify(0.1))   

    net['conv23/3x3_s1'] = ConvLayer( net['conv22/3x3_s2'], 1024, 3, stride=1, pad=1, flip_filters=False, nonlinearity=LeakyRectify(0.1))       
    net['conv24/3x3_s1'] = ConvLayer( net['conv23/3x3_s1'], 1024, 3, stride=1, pad=1, flip_filters=False, nonlinearity=LeakyRectify(0.1))   

    # dense layer
    net['dense1'] = DenseLayer(net['conv24/3x3_s1'], num_units=4096, nonlinearity = LeakyRectify(0.1))
    net['dropout'] = DropoutLayer(net['dense1'], p=0.5)
    net['dense2'] = DenseLayer(net['dropout'], num_units=1470, nonlinearity=linear)

    net['output_layer'] = net['dense2']    
    ## detection params

    return net
