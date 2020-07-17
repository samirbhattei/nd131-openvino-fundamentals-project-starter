#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.ie_network = None
        self.loaded_model = None
        self.model_input = None
        self.model_output = None

    def load_model(self, data):
        ### TODO: Load the model ###
        xml_model = data["model"]
        bin_model = os.path.splitext(xml_model)[0] + ".bin"
        
        ie_engine = IECore()
        self.ie_network = IENetwork(model=xml_model, weights=bin_model)
        ### TODO: Check for supported layers ###
        layers_supported = ie_engine.query_network(self.ie_network, device_name='CPU')
        layers = self.ie_network.layers.keys()
        flag_unsupported_layer = None
        for l in layers:
            if l not in layers_supported:
                flag_unsupported_layer = True
            
        ### TODO: Add any necessary extensions ###
        if flag_unsupported_layer: 
            ie_engine.add_extension(data["cpu_extension"], data["device"])
            
        self.loaded_model = ie_engine.load_network(self.ie_network, data["device"])
        self.model_input = next(iter(self.ie_network.inputs))
        self.model_output = next(iter(self.ie_network.outputs))
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return self.model_input, self.model_output

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        input_shapes = {}
        for inp in self.ie_network.inputs:
            input_shapes[inp] = (self.ie_network.inputs[inp].shape)
        return input_shapes

    def exec_net(self, input_dict, request):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        inference_request_handler = self.loaded_model.start_async(
                request_id=request, 
                inputs=input_dict)
        ### Note: You may need to update the function parameters. ###
        return inference_request_handler

    def wait(self, inference_request):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = inference_request.wait()
        return status

    def get_output(self, inference_request, output):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        out = inference_request.outputs[output]
        return out

