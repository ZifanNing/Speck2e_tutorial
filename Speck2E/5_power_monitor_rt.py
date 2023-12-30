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
input_shape = (2, 128, 128)

cnn = nn.Sequential(
    # 2 x 128 x 128
    # Core 0
    nn.Conv2d(2, 8, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False),  # 8, 64, 64
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=(2, 2)),  # 8,32,32
    # """Core 1"""
    nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 16, 32, 32
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=(2, 2)),  # 16, 16, 16
    # """Core 2"""
    nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 8, 16, 16
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=(2, 2)),  # 16, 16, 16
    nn.Flatten(),
    nn.Dropout2d(0.5),
    nn.Linear(512, 2, bias=False),
    nn.ReLU(),
    )

# assign a handcraft weight to CNN
# cnn[0].weight.data = torch.ones_like(cnn[0].weight.data, dtype=torch.float32) * 0.05

# cnn to snn
snn = from_model(cnn, input_shape=input_shape, batch_size=1).spiking_model

# snn to DynapcnnNetwork
dynapcnn_net = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=True)

dynapcnn_device_str = "speck2edevkit:0"
devkit_cfg = dynapcnn_net.make_config(device=dynapcnn_device_str, monitor_layers=["dvs"])


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

# init samna, endpoints should correspond to visualizer, if some port is already bound, please change it.
samna_node = samna.init_samna()
time.sleep(1)   # wait tcp connection build up, this is necessary to open remote node.



dk = samna.device.open_device("Speck2eDevKit:0")
power_monitor = dk.get_power_monitor()

# create samna node for power reading
power_source_node = power_monitor.get_source_node()
power_buffer_node = samna.BasicSinkNode_unifirm_modules_events_measurement()


# route events
dvs_graph = samna.graph.EventFilterGraph()
# build_samna_event_route(dk, dvs_graph)
_, _, streamer = dvs_graph.sequential([dk.get_model_source_node(), "Speck2eDvsToVizConverter", "VizEventStreamer"])
dvs_graph.sequential([power_source_node, power_buffer_node])


dvs_graph.sequential([power_source_node, "MeasurementToVizConverter", streamer])
streamer.set_streamer_endpoint("tcp://0.0.0.0:40000")

print(dvs_graph)

dvs_graph.start()

visualizer_id = 3
visualizer, gui_thread = open_visualizer(0.75, 0.75, samna_node.get_receiver_endpoint(), samna_node.get_sender_endpoint(), visualizer_id)

activity_plot_id = visualizer.plots.add_activity_plot(128, 128, "DVS Layer")
plot_name = "plot_" + str(activity_plot_id)
plot = getattr(visualizer, plot_name)
plot.set_layout(0, 0, 0.6, 1)   # set the position: top left x, top left y, bottom right x, bottom right y




visualizer.receiver.set_receiver_endpoint("tcp://0.0.0.0:40000")
visualizer.receiver.add_destination(visualizer.splitter.get_input_channel())
visualizer.splitter.add_destination("passthrough", visualizer.plots.get_plot_input(activity_plot_id))

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

# visualizer.plots.report()
# print("Now you should see a change on the GUI window!")


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