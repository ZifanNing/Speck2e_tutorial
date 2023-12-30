# Suppress warnings (This is only to keep the notebook pretty. You might want to comment the below two lines)
import warnings

warnings.filterwarnings("ignore")

# - Import statements
import torch
import samna
import time
from tqdm.auto import tqdm
import numpy as np
import torch.nn as nn
from torchvision import datasets
import sinabs
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import io
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from sinabs.layers.pool2d import SumPool2d
import time
from threading import Thread
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from multiprocessing import Process
from typing import Union
import random

assert samna.__version__ >= '0.21.8', f"samna version {samna.__version__} is too low for this experiment"

# create a one layer CNN
input_shape = (1, 32, 32)

cnn = nn.Sequential(SumPool2d(kernel_size=(1, 1)),
                    nn.Conv2d(in_channels=1,
                              out_channels=2,
                              kernel_size=(16, 16),
                              stride=(1, 1),
                              padding=(0, 0),
                              bias=False),
                    nn.ReLU())

# set handcraft weights for the CNN
weight_ones = torch.ones(1, 8, 16, dtype=torch.float32)
weight_zeros = torch.zeros(1, 8, 16, dtype=torch.float32)

channel_1_weight = torch.cat([weight_ones, weight_zeros], dim=1).unsqueeze(0)
channel_2_weight = torch.cat([weight_zeros, weight_ones], dim=1).unsqueeze(0)
handcraft_weight = torch.cat([channel_1_weight, channel_2_weight], dim=0)

output_cnn_lyr_id = 1
cnn[output_cnn_lyr_id].weight.data = handcraft_weight


# cnn to snn
snn = from_model(cnn, input_shape=input_shape, batch_size=1).spiking_model

# snn to DynapcnnNetwork
dynapcnn_net = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=True)

dynapcnn_device_str = "speck2edevkit:0"
devkit_cfg = dynapcnn_net.make_config(device=dynapcnn_device_str, monitor_layers=["dvs"])

cnn_output_layer = dynapcnn_net.chip_layers_ordering[-1]
devkit_cfg.cnn_layers[cnn_output_layer].monitor_enable = True

"""dvs layer configuration"""
# link the dvs layer to the 1st layer of the cnn layers
devkit_cfg.dvs_layer.destinations[0].enable = True
devkit_cfg.dvs_layer.destinations[0].layer = dynapcnn_net.chip_layers_ordering[0]
# merge the polarity of input events
devkit_cfg.dvs_layer.merge = True
# drop the raw input events from the dvs sensor, since we write events to devkit manually
devkit_cfg.dvs_layer.pass_sensor_events = False
# enable monitoring the output from dvs pre-preprocessing layer
devkit_cfg.dvs_layer.monitor_enable = True

# def build_samna_event_route(dk, dvs_graph):
#     # build a graph in samna to show dvs
#     _, _, streamer = dvs_graph.sequential([dk.get_model_source_node(), "Speck2eDvsToVizConverter", "VizEventStreamer"])

#     streamer.set_streamer_endpoint("tcp://0.0.0.0:40000")


def open_visualizer(window_width, window_height, receiver_endpoint, sender_endpoint, visualizer_id):
    # start visualizer in a isolated process which is required on mac, intead of a sub process.
    # it will not return until the remote node is opened. Return the opened visualizer.
    gui_cmd = '''%s -c "import samna, samnagui; samnagui.runVisualizer(%f, %f, '%s', '%s', %d)"''' % \
        (sys.executable, window_width, window_height, receiver_endpoint, sender_endpoint, visualizer_id)
    print("Visualizer start command: ", gui_cmd)
    gui_thread = Thread(target=os.system, args=(gui_cmd,))
    gui_thread.start()

    # wait for open visualizer and connect to it.
    timeout = 10
    begin = time.time()
    name = "visualizer" + str(visualizer_id)
    while time.time() - begin < timeout:
        try:
            time.sleep(0.05)
            samna.open_remote_node(visualizer_id, name)
        except:
            continue
        else:
            return getattr(samna, name), gui_thread

    raise Exception("open_remote_node failed:  visualizer id %d can't be opened in %d seconds!!" % (visualizer_id, timeout))

def create_fake_input_events(time_sec: int, data_rate: int = 1000):

    """
    Args:
        time_sec: how long is the input events
        data_rate: how many input events generated in 1 second

        During the first half time, it generates events where y coordinate only in range[0, 7] which means top half
        region of the input feature map.

        Then in the last half of time, it generates events where y coordinate only in range[8, 15] which means bottom
        half region of the input feature map.

    """

    time_offset_micro_sec = 5000  # make the timestamp start from 5000
    time_micro_sec = time_sec * 1000000  # timestamp unit is micro-second
    time_stride = 1000000 // data_rate

    half_time = time_micro_sec // 2

    events = []
    for time_stamp in range(time_offset_micro_sec, time_micro_sec + time_offset_micro_sec + 1, time_stride):

        spk = samna.speck2e.event.DvsEvent()
        spk.timestamp = time_stamp
        spk.p = random.randint(0, 1)
        spk.x = random.randint(0, 15)

        if time_stamp < half_time:
            spk.y = random.randint(0, 7)  # spike located in top half of the input region
        else:
            spk.y = random.randint(8, 15)  # spike located in bottom half of the input region

        events.append(spk)

    return events


# init samna, endpoints should correspond to visualizer, if some port is already bound, please change it.
samna_node = samna.init_samna()
time.sleep(1)   # wait tcp connection build up, this is necessary to open remote node.



dk = samna.device.open_device("Speck2eDevKit:0")
power_monitor = dk.get_power_monitor()


# init necessary nodes in samna graph
# node for writing fake inputs into devkit
input_buffer_node = samna.BasicSourceNode_speck2e_event_speck2e_input_event()
# node for reading Spike(i.e. the output from last CNN layer)
spike_buffer_node = samna.BasicSinkNode_speck2e_event_output_event()
# create samna node for power reading
power_source_node = power_monitor.get_source_node()
power_buffer_node = samna.BasicSinkNode_unifirm_modules_events_measurement()


# route events
dvs_graph = samna.graph.EventFilterGraph()
# build_samna_event_route(dk, dvs_graph)
_, _, streamer = dvs_graph.sequential([dk.get_model_source_node(), "Speck2eDvsToVizConverter", "VizEventStreamer"])
# branch #2: for the spike count plot (first divide spike events into groups by class, then count spike events per class)
_, spike_collection_filter, spike_count_filter, _ = dvs_graph.sequential(
    [dk.get_model_source_node(), "Speck2eSpikeCollectionNode", "Speck2eSpikeCountNode", streamer])
# branch #3: for obtaining the output Spike from cnn output layer
_, type_filter_node_spike, _ = dvs_graph.sequential(
    [dk.get_model_source_node(), "Speck2eOutputEventTypeFilter", spike_buffer_node])
dvs_graph.sequential([power_source_node, power_buffer_node])


dvs_graph.sequential([power_source_node, "MeasurementToVizConverter", streamer])
streamer.set_streamer_endpoint("tcp://0.0.0.0:40000")
# add desired type for filter node
type_filter_node_spike.set_desired_type("speck2e::event::Spike")
# add configurations for spike collection and counting filters
time_interval = 50
labels = ["0", "1"]  # a list that contains the names of output classes
num_of_classes = len(labels)
spike_collection_filter.set_interval_milli_sec(time_interval)  # divide according to this time period in milliseconds.
spike_count_filter.set_feature_count(num_of_classes)  # number of output classes


print(dvs_graph)

dvs_graph.start()

visualizer_id = 3
visualizer, gui_thread = open_visualizer(0.75, 0.75, samna_node.get_receiver_endpoint(), samna_node.get_sender_endpoint(), visualizer_id)

activity_plot_id = visualizer.plots.add_activity_plot(128, 128, "DVS Layer")
plot_name = "plot_" + str(activity_plot_id)
plot = getattr(visualizer, plot_name)
plot.set_layout(0, 0, 0.6, 1)   # set the position: top left x, top left y, bottom right x, bottom right y

# add spike count plot to gui
spike_count_id = visualizer.plots.add_spike_count_plot("Spike Count", num_of_classes, labels)
plot = visualizer.plot_1
plot.set_layout(0.5, 0.5, 1, 1)
plot.set_show_x_span(10)  # set the range of x axis
plot.set_label_interval(1.0)  # set the x axis label interval
plot.set_max_y_rate(1.2)  # set the y axis max value according to the max value of all actual values. 
plot.set_show_point_circle(True)  # if show a circle of every point.
plot.set_default_y_max(10)  # set the default y axis max value when all points value is zero.


visualizer.receiver.set_receiver_endpoint("tcp://0.0.0.0:40000")
visualizer.receiver.add_destination(visualizer.splitter.get_input_channel())
visualizer.splitter.add_destination("passthrough", visualizer.plots.get_plot_input(activity_plot_id))
visualizer.splitter.add_destination("spike_count", visualizer.plots.get_plot_input(spike_count_id))
visualizer.plots.report()

power_plot_id = visualizer.plots.add_power_measurement_plot("power consumption", 5, ["io", "ram", "logic", "vddd", "vdda"])
plot_name = "plot_" + str(power_plot_id)
print("plot_name: ", plot_name)
plot = getattr(visualizer, plot_name)
plot.set_layout(0, 0.75, 1.0, 1.0)
plot.set_show_x_span(10)
plot.set_label_interval(2)
plot.set_max_y_rate(1.5)
plot.set_show_point_circle(False)
plot.set_default_y_max(1)
plot.set_y_label_name("power (mW)")  # set the label of y axis
visualizer.splitter.add_destination("measurement", visualizer.plots.get_plot_input(power_plot_id))

# create fake input events
input_time_length = 5 # seconds
data_rate = 5000
input_events = create_fake_input_events(time_sec=input_time_length, data_rate=data_rate)

print(f"number of fake input spikes: {len(input_events)}")

# apply the config to devkit
dk.get_model().apply_configuration(devkit_cfg)
time.sleep(0.1)

# write the fake input into the devkit

# enable & reset the stop-watch of devkit, this is mainly for the timestamp processing for the input&output events.
stop_watch = dk.get_stop_watch()
stop_watch.set_enable_value(True)
stop_watch.reset()
time.sleep(0.01)

# clear output buffer
# spike_buffer_node.get_events()

# write through the input buffer node
input_time_length = (input_events[-1].timestamp - input_events[0].timestamp) / 1e6
input_buffer_node.write(input_events)
# sleep till all input events is sent and processed
time.sleep(input_time_length + 0.02)

# get the output events from last DynapCNN Layer
dynapcnn_layer_events = spike_buffer_node.get_events()

print("You should see the input events through the GUI window as well as the spike count of output!")

# visualizer.plots.report()
# print("Now you should see a change on the GUI window!")

print(f"number of fake input spikes: {len(input_events)}")
print(f"number of output spikes from DynacpCNN Layer: {len(dynapcnn_layer_events)}")

# get the timestamp of the output event
spike_timestamp = [each.timestamp for each in dynapcnn_layer_events]
print(len(spike_timestamp))
# shift timestep starting from 0
start_t = spike_timestamp[0]
spike_timestamp = [each - start_t for each in spike_timestamp]

# get the neuron index of each output spike 
neuron_id = [each.feature  for each in dynapcnn_layer_events]


# plot the output neuron index vs. time
fig, ax = plt.subplots()
ax.scatter(spike_timestamp, neuron_id)
ax.set(xlim=(0, input_time_length * 1e6),ylim=(-0.5, 1.5))
ax.set_xlabel("time( micro sec)")
ax.set_ylabel("neuron index")
ax.set_title("OutputSpike")


# modify configuration
config = samna.speck2e.configuration.SpeckConfiguration()
# enable dvs event monitoring
devkit_cfg.dvs_layer.monitor_enable = True
dk.get_model().apply_configuration(devkit_cfg)
time.sleep(0.1)

sample_rate = 100

power_monitor.start_auto_power_measurement(sample_rate)
print(power_monitor)

# wait until visualizer window destroys.
gui_thread.join()

dvs_graph.stop()
samna.device.close_device(dk)