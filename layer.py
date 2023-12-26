import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import os


class RoIPoolingLayer(Layer):
    def __init__(self, _height_, _width_, **kwargs):
        """
        params:
            _height_: int
            _width_: int
        """
        self._height_ = _height_
        self._width_ = _width_
        
        super(RoIPoolingLayer, self).__init__(**kwargs)


    def call(self, x):
        """ Maps the input tensor of the ROI layer to its output
        this function computes the pooled area
        the element in 0th index in the parameter x is feature map tensor with shape (batch_size, _height_, _width_, n_channels)
        and the element in 1th index in the parameter x is roi bounding box items with shape (batch_size, num_rois, 4)
        each region of interest is defined by four relative coordinates (x_min, y_min, x_max, y_max) between 0 and 1
        """
        def pool_rois(x): 
          return RoIPoolingLayer.rois_pooling(x[0], x[1], self._height_, self._width_)
        
        return tf.map_fn(pool_rois, x, dtype=tf.float32)


    @staticmethod
    def rois_pooling(feature_map, rois, _height_, _width_):
        """ Applies ROI pooling for a single image and varios ROIs
        params:
            feature_map: array
            rois: array
            _height_: int
            _width_: int
        """
        def pool_roi(roi): 
          return RoIPoolingLayer.roi_pooling(feature_map, roi, _height_, _width_)
        
        return tf.map_fn(pool_roi, rois, dtype=tf.float32)
        
    
    @staticmethod
    def roi_pooling(feature_map, roi, _height_, _width_):
        """
        apply roi pooling to a single mapped roi
        params:
            feature_map: array
            roi: array
            _height_: int
            _width_: int

        return:
            pooled_features: array
        """

        # compute roi    
        feature_map_height = int(feature_map.shape[0])
        feature_map_width  = int(feature_map.shape[1])
        
        h_start = tf.cast(feature_map_height * roi[0], 'int32')
        w_start = tf.cast(feature_map_width  * roi[1], 'int32')
        h_end   = tf.cast(feature_map_height * roi[2], 'int32')
        w_end   = tf.cast(feature_map_width  * roi[3], 'int32')
        
        region = feature_map[h_start:h_end, w_start:w_end, :]
        
        # divide the region into non overlapping areas
        region_height = h_end - h_start
        region_width  = w_end - w_start
        h_step = tf.cast( region_height / _height_, 'int32')
        w_step = tf.cast( region_width  / _width_ , 'int32')
        
        areas = [[(
                    i*h_step, 
                    j*w_step, 
                    (i+1)*h_step if i+1 < _height_ else region_height, 
                    (j+1)*w_step if j+1 < _width_ else region_width
                   ) 
                   for j in range(_width_)] 
                  for i in range(_height_)]
        
        # take the maximum of each area and stack the result
        def pool_area(x): 
          return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])
        
        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
        return pooled_features