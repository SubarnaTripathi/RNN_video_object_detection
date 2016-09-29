# RNN_video_object_detection

This is the implementation of our BMVC 2016 paper "Context Matters: Refining Object Detection in Video with Recurrent Neural Networks".

1. It includes conversion of YOLO binary weight file to python weights
2. finetuning YOLO with theano and lasagne for youtube-objects dataset : DA-YOLO as referred in the paper, the pseudo-label generator
3. I have included the compatible label/annotations I needed to create for training & eval in theano.
4. Pre-calculated features for training and test numpy arrays are available with me. However, those are bigger than the allowable file size for uploading.
5. fine tuned DA-YOLO weights are 1.1 GB, however the conversion python file could be used to generate those. YOLO weight bin file can be downloaded from darknet website. 
6. GRU training and evaluation code and visual results - this notebook is large. 
   use the following command if it doesn't open directly here:
   http://nbviewer.jupyter.org/ and use SubarnaTripathi/RNN_video_object_detection in the box. And, open   
                         RNN_Object_Detection_GRU_Smoothness_visual_results.ipynb


** The code requires cleaning, which I'll eventually do. **
