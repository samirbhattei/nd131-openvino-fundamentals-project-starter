"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client =  mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def validate_input_file(input_data):
    if not os.path.isfile(input_data):
        return 0, None
    supported_video_formats = [
        ".mp4", ".avi", ".mov", ".mpeg", ".flv", ".wmv"]
    supported_image_formats = [
        ".jpg", ".jpeg", ".bmp", ".png", ".tiff"]
    _, f_extension = os.path.splitext(input_data)
    if f_extension.lower() in supported_image_formats:
        return 1, 'image'
    elif f_extension.lower() in supported_video_formats:
        return 1, 'video'
    else:
        return 1, None

def map_coords(points, image):
    mapped_points = []
    mapped_points.append(int(points[0] * image.shape[1]))
    mapped_points.append(int(points[1] * image.shape[0]))
    mapped_points.append(int(points[2] * image.shape[1]))
    mapped_points.append(int(points[3] * image.shape[0]))
    return mapped_points

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    network_object = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    data = {}
    data["model"] = args.model
    data["cpu_extension"] = args.cpu_extension
    data["device"] = args.device
    data["input"] = args.input
    data["threshold"] = prob_threshold
    ### TODO: Load the model through `infer_network` ###
    m_input, m_output = network_object.load_model(data)
    net_in_shape = network_object.get_input_shape()
    n, channel, height, width = net_in_shape['image_tensor']
    ### TODO: Handle the input stream ###

    valid_data, data_type = validate_input_file(args.input)
    if not valid_data:
        return
    if valid_data and not data_type:
        return
    count = 0
    total = 0
    time_duration = None
    temp_count = 0
    ### TODO: Loop until stream is over ###
    if valid_data and data_type == 'video':
        vid_cap = cv2.VideoCapture(args.input)
        vid_cap.open(args.input)


        ### TODO: Read from the video capture ###
        while True:
            if not vid_cap.isOpened():
                break
            _ret, img_frame = vid_cap.read()
            if not _ret:
                break
            st_time = time.time()
            ### TODO: Pre-process the image as needed ###
            image = cv2.resize(img_frame, (net_in_shape['image_tensor'][3], net_in_shape['image_tensor'][2]))
            processed_image = image.transpose((2, 0, 1))
            processed_image = processed_image.reshape(1, *processed_image.shape)

            ### TODO: Start asynchronous inference for specified request ###
            request_handler = network_object.exec_net({m_input: processed_image}, 0)

            ### TODO: Wait for the result ###
            if network_object.wait(request_handler) == 0:

                ### TODO: Get the results of the inference request ###
                result = network_object.get_output(request_handler, m_output)
                ### TODO: Extract any desired stats from the results ###
                data_points = result[0][0]
                count = 0
                for list_item in data_points:
                    current_prob = list_item[2]
                    current_bbox = []
                    if current_prob >= data["threshold"]:
                        current_bbox.append(list_item[3])
                        current_bbox.append(list_item[4])
                        current_bbox.append(list_item[5])
                        current_bbox.append(list_item[6])
                        img_points = map_coords(current_bbox, img_frame)
                        cv2.rectangle(img_frame, (img_points[0], img_points[1]), (img_points[2], img_points[3]), (255, 0, 255), 1)
                        ### TODO: Calculate and send relevant information on ###
                        count = count + 1
                        str_msg = "Person " + str(count)
                        cv2.putText(img_frame,str_msg,(img_points[0]-10, img_points[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
                ### current_count, total_count and duration to the MQTT server ###
                if count > temp_count: # New entry
                    total = total + count - temp_count
                    client.publish('person', payload=json.dumps({'total': total}))
                ### Topic "person": keys of "count" and "total" ###
                client.publish('person', payload=json.dumps({'count': count}))
                if count < temp_count:
                    ### Topic "person/duration": key of "duration" ###
                    time_duration = (time.time() - st_time) * 1000
                    client.publish('person/duration',
                                   payload=json.dumps({'duration': time_duration}))

                temp_count = count
            ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(img_frame)
            sys.stdout.flush()

        vid_cap.release()
    ### TODO: Write an output image if `single_image_mode` ###
    elif valid_data and data_type == 'image':
        img_frame = cv2.imread(args.input)
        st_time = time.time()

        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(img_frame, (net_in_shape['image_tensor'][3], net_in_shape['image_tensor'][2]))
        processed_image = image.transpose((2, 0, 1))
        processed_image = processed_image.reshape(1, *processed_image.shape)

        ### TODO: Start asynchronous inference for specified request ###
        request_handler = network_object.exec_net({'image_tensor': processed_image}, 0)

        ### TODO: Wait for the result ###
        if network_object.wait(request_handler) == 0:

            ### TODO: Get the results of the inference request ###
            result = network_object.get_output(request_handler, m_output)

            ### TODO: Extract any desired stats from the results ###
            data_points = result[0][0]
            count = 0
            for list_item in data_points:
                current_prob = list_item[2]
                current_bbox = []
                if current_prob >= data["threshold"]:
                    current_bbox.append(list_item[3])
                    current_bbox.append(list_item[4])
                    current_bbox.append(list_item[5])
                    current_bbox.append(list_item[6])
                    img_points = map_coords(current_bbox, img_frame)
                    cv2.rectangle(img_frame, (img_points[0], img_points[1]), (img_points[2], img_points[3]), (255, 0, 255), 1)
                    ### TODO: Calculate and send relevant information on ###
                    count = count + 1
                    str_msg = "Person " + str(count)
                    cv2.putText(img_frame,str_msg,(img_points[0]-10, img_points[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
            ### current_count, total_count and duration to the MQTT server ###
            if count > temp_count: # New entry
                total = total + count - temp_count
                client.publish('person', payload=json.dumps({'total': total}))
            ### Topic "person": keys of "count" and "total" ###
            client.publish('person', payload=json.dumps({'count': count}))
            if count < temp_count:
                ### Topic "person/duration": key of "duration" ###
                time_duration = (time.time() - st_time) * 1000
                client.publish('person/duration',
                                payload=json.dumps({'duration': time_duration}))

            temp_count = count
            ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(img_frame)
            sys.stdout.flush()


    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
