from finite_state_machine import ProximityDetectionFSM as FSM

from addict import Dict as ADDict
import logging
import numpy as np
import os
from PySide2.QtCore import QObject, Signal, Slot, QThread
from PySide2.QtWidgets import QWidget, QApplication, QLabel
from PySide2.QtGui import QPixmap
import samna
from sinabs.layers import IAFSqueeze
from sinabs.backend.dynapcnn import DynapcnnNetwork
import sys
from threading import Thread
import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union


# ------------------------ config ---------------------------------------------------------
# log
fmt = "[%(asctime)s] %(name)s %(levelname)s %(lineno)d: %(message)s"
datefmt = "%m/%d %H:%M:%S"
logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG, format=fmt, datefmt=datefmt
)


DECIMATOR = {
    '1_1': False,  # means no decimator
    '1_2': 0b000,
    '1_4': 0b001,
    '1_8': 0b010,
    '1_16': 0b011,
    '1_32': 0b100,
}


proximity_detection_net = nn.Sequential(
    *[
        # [B, 1, 128, 128] -> [B, 8, 32, 32]
        nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2, 2), stride=(2, 2), bias=False),
        IAFSqueeze(batch_size=1, min_v_mem=-1),
        nn.AvgPool2d(kernel_size=(2, 2)),

        # [B, 8, 32, 32] -> [B, 32, 16, 16]
        nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias=False),
        IAFSqueeze(batch_size=1, min_v_mem=-1),
        nn.AvgPool2d(kernel_size=(2, 2)),

        # [B, 32, 16, 16] -> [B, 4, 16, 16]
        nn.Conv2d(in_channels=32, out_channels=4, kernel_size=(3, 3), padding=(1, 1), bias=False),
        IAFSqueeze(batch_size=1, min_v_mem=-1),

        # [B, 1024] -> [B, 5]  # 
        nn.Flatten(),
        nn.Linear(1024, 5, bias=False),
        IAFSqueeze(batch_size=1, min_v_mem=-1),
    ]
)


proximity_detection_cfg = {
    'MODEL': {
        'INPUT_SHAPE': [1, 128, 128],
        'NUM_CLASSES': 5,
        'NET': proximity_detection_net,
        'WEIGHTS': '8_demo/weights/2022-11-21_09-49-09_epoch=99.pth'
    },
    'DYNAPCNN': {
        'READOUT_LAYER_ID': 12,
        'THRESHOLD': 10,
        'TIME_INTERVAL': 0.2,  # seconds
        'SLOW_CLK_RATE': 10,  # Hz
        'DECIMATOR': '1_1',
        'DVS_FILTER': False,  # speck2f doesnot support dvs filter
        'LOW_PASS_FILTER': False,  # set low pass filter, unhelpful for proximity detection
    },
    'POWER': {
        'FREQUENCY': 10,
        'CHANNELS': ['VDD_IO', 'VDD_RAM', 'VDD_LOGIC', 'VDD_PIXEL_DIGITAL', 'VDD_PIXEL_ANALOG'],
        'WRITE_TO_DISK': False,
        'SAVE_FILE_NAME': 'online_power_consumption',
    },
    'RESULT_IMGS': {
        0: '8_demo/imgs/silent.png',
        1: '8_demo/imgs/Near3m.png',
        2: '8_demo/imgs/Near2m.png',
        3: '8_demo/imgs/Near1m.png',
        4: '8_demo/imgs/Away2m.png',
        5: '8_demo/imgs/Away3m.png',
        6: '8_demo/imgs/left.png',
        7: '8_demo/imgs/right.png',
    }
}

cfg = ADDict(proximity_detection_cfg)
# ------------------------ config ---------------------------------------------------------


def read_icons() -> Dict[str, np.ndarray]:
    # load icons
    icons = {}

    h, w = 480, 480

    for result, img_path in cfg.RESULT_IMGS.items():
        icons[result] = QPixmap(img_path).scaled(w, h)

    return icons


def adjust_net_for_speck2e() -> nn.Module:
    net = cfg.MODEL.NET
    linear = net[-2]
    in_features, out_features, bias = linear.in_features, linear.out_features, linear.bias

    # due to the readout mapping of speck2e
    new_out_features = (out_features - 1) * 4 + 1
    new_linear = nn.Linear(in_features, new_out_features, bias=bias)
    net[-2] = new_linear
    return net


def adjust_weight_for_speck2e(weight: torch.Tensor):
    out_features, in_features = weight.shape

    new_out_features = (out_features - 1) * 4 + 1

    new_w = torch.zeros(new_out_features, in_features)

    for inx, w in enumerate(weight):
        new_w[inx * 4, :] = weight[inx, :]

    return new_w


def load_model() -> nn.Module:
    '''
    Define model and load weights to model.
    '''
    weights_file = cfg.MODEL.WEIGHTS
    assert os.path.exists(weights_file), f'Weights not exists! {weights_file}'

    # load weights
    checkpoint = torch.load(weights_file, map_location=torch.device('cpu'))

    # load model
    snn = adjust_net_for_speck2e()
    logging.info(f'snn: {snn}')
    weights = snn.state_dict()

    for name, weight in checkpoint.items():
        if 'weight' in name and name in weights:
            if weight.shape == weights[name].shape:
                logging.info(f'same weight shape in {name}')
                weights[name] = weight
            else:
                logging.info(f'adjust weight in {name}')
                weights[name] = adjust_weight_for_speck2e(weight)

    snn.load_state_dict(weights, strict=True)

    return snn


def open_speck2e():
    ''' sinabs library somehow wraps open device. '''
    # return samna.device.open_device("Speck2eTestBoard:0")
    return samna.device.open_device("Speck2eDevKit:0")


def open_visualizer(window_width, window_height, receiver_endpoint, sender_endpoint, visualizer_id):
    # start visualizer in a isolated process which is required on mac, intead of a sub process.
    # it will not return until the remote node is opened. Return the opened visualizer.
    gui_cmd = f"import samna, samnagui; samnagui.runVisualizer({window_width}, {window_height}, '{receiver_endpoint}', '{sender_endpoint}', {visualizer_id})"
    os_cmd = f'{sys.executable} -c "{gui_cmd}"'
    logging.info(f"Visualizer start command: {os_cmd}")
    gui_thread = Thread(target=os.system, args=(os_cmd,))
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

    raise Exception(f"open_remote_node failed:  visualizer id({visualizer_id}) can't be opened in {timeout} seconds!!")


def model_config():
    snn = load_model()
    snn.eval()

    dynapcnn_model = DynapcnnNetwork(
        snn, input_shape=cfg.MODEL.INPUT_SHAPE, dvs_input=True, discretize=True,
    )

    config = dynapcnn_model.make_config(chip_layers_ordering='auto', device='speck2e:0')
    logging.info(f'{dynapcnn_model.chip_layers_ordering}')

    chip_layers_ordering = dynapcnn_model.chip_layers_ordering

    output_layer_id = chip_layers_ordering[-1]
    logging.info(f'output_layer_id: {output_layer_id}')
    config.dvs_layer.monitor_enable = True
    config.cnn_layers[output_layer_id].monitor_enable = True
    config.cnn_layers[output_layer_id].destinations[0].enable = True
    config.cnn_layers[output_layer_id].destinations[0].layer = cfg.DYNAPCNN.READOUT_LAYER_ID

    decimator = DECIMATOR[cfg.DYNAPCNN.DECIMATOR]
    if decimator:
        # speck2e only supports to set decimator at the output of cnn layer
        # set decimator to the output of first cnn layer
        first_cnn_layer_inx = chip_layers_ordering[0]
        config.cnn_layers[first_cnn_layer_inx].output_decimator_enable = True
        config.cnn_layers[first_cnn_layer_inx].output_decimator_interval = decimator

    # decrease power consumption of VDD_PIXEL_DIGITAL and VDD_PIXEL_ANALOG
    config.factory_config.dvs_layer.current_out1 = 2
    config.factory_config.dvs_layer.current_control_p0 = 13
    config.factory_config.dvs_layer.current_out2 = 5
    config.factory_config.dvs_layer.current_out3 = 5
    config.factory_config.dvs_layer.current_control_p3 = 0
    config.factory_config.dvs_layer.current_control_p4 = 0
    config.factory_config.dvs_layer.current_control_p5 = 30
    config.factory_config.dvs_layer.current_control_p6 = 13

    if cfg.DYNAPCNN.DVS_FILTER:
        # set dvs flter
        config.dvs_filter.enable = True
        config.factory_config.dvs_filter.filter_clk_enable = True

    # set readout layer
    config.readout.enable = True
    config.readout.readout_configuration_sel = 0b11
    # config.readout.output_mod_sel = 0b10  # useless

    # If the spike num of any class in readout layer is larger than threshold, then chip emits an interupt.
    # And the interupt contains that class index.
    config.readout.threshold = cfg.DYNAPCNN.THRESHOLD

    if cfg.DYNAPCNN.LOW_PASS_FILTER:
        # Selects readout moving average length
        config.readout.low_pass_filter_disable = False  # True for length = 1, False for setting low_pass_filter32_not16
        config.readout.low_pass_filter32_not16 = False  # False for length = 16, True for length = 32
    else:
        config.readout.low_pass_filter_disable = True

    # allow read pin value
    # Once the fpga on speck2e chip receive the interupt from readout layer, then it generates pin value.
    config.readout.readout_pin_monitor_enable = True

    # open device
    devkit = open_speck2e()

    # start running on hardware
    devkit.get_model().apply_configuration(config)

    # set timestamp
    stopWatch = devkit.get_stop_watch()
    stopWatch.set_enable_value(True)

    # set io of the devkit
    dk_io = devkit.get_io_module()

    # slow clock frequency on chip
    # The frequency of speck2e chip to process one batch of data is 10 Hz, and the frequency cannot be changed.
    slow_clock_rate = cfg.DYNAPCNN.SLOW_CLK_RATE
    logging.info(f'slow_clock_rate: {slow_clock_rate} Hz')
    dk_io.set_slow_clk_rate(slow_clock_rate)  # Hz
    dk_io.set_slow_clk(True)

    return config, devkit


def samna_initialization(devkit):
    # init samna, endpoints should correspond to visualizer, if some port is already bound, please change it.
    samna_node = samna.init_samna()
    sender_endpoint = samna_node.get_sender_endpoint()
    receiver_endpoint = samna_node.get_receiver_endpoint()
    time.sleep(1)   # wait tcp connection build up, this is necessary to open remote node.

    # get power monitor
    power = devkit.get_power_monitor()

    visualizer_id = 3
    visualizer, gui_thread = open_visualizer(0.421875, 0.75, receiver_endpoint, sender_endpoint, visualizer_id)

    streamer_endpoint = 'tcp://0.0.0.0:40000'
    # set visualizer's receiver endpoint to streamer's sender endpoint
    visualizer.receiver.set_receiver_endpoint(streamer_endpoint)
    # connect the receiver output to splitter inside the visualizer
    visualizer.receiver.add_destination(visualizer.splitter.get_input_channel())

    # add plots to gui
    # # add dvs plot
    activity_plot_id = visualizer.plots.add_activity_plot(128, 128, "DVS Layer")
    plot_name = "plot_" + str(activity_plot_id)
    plot = getattr(visualizer, plot_name)
    plot.set_layout(0, 0, 0.75, 0.75)   # set the position: top left x, top left y, bottom right x, bottom right y
    visualizer.splitter.add_destination("passthrough", visualizer.plots.get_plot_input(activity_plot_id))
    # # add power measurement plot
    power_plot_id = visualizer.plots.add_power_measurement_plot("power consumption", 5, ["io", "ram", "logic", "vddd", "vdda"])
    plot_name = "plot_" + str(power_plot_id)
    plot = getattr(visualizer, plot_name)
    plot.set_layout(0, 0.75, 1.0, 1.0)
    plot.set_show_x_span(10)
    plot.set_label_interval(2)
    plot.set_max_y_rate(1.5)
    plot.set_show_point_circle(False)
    plot.set_default_y_max(1)
    plot.set_y_label_name("power (mW)")  # set the label of y axis
    visualizer.splitter.add_destination("measurement", visualizer.plots.get_plot_input(power_plot_id))
    visualizer.plots.report()

    graph = samna.graph.EventFilterGraph()
    pin_buffer_node = samna.BasicSinkNode_speck2e_event_output_event()
    power_buffer_node = samna.BasicSinkNode_unifirm_modules_events_measurement()

    # init the graph
    _, _, streamer = graph.sequential([devkit.get_model_source_node(), "Speck2eDvsToVizConverter", "VizEventStreamer"])
    _, type_filter_node_pin, _ = graph.sequential([devkit.get_model_source_node(), "Speck2eOutputEventTypeFilter", pin_buffer_node])
    graph.sequential([power.get_source_node(), "MeasurementToVizConverter", streamer])
    graph.sequential([power.get_source_node(), power_buffer_node])

    # set the important nodes of the graph
    streamer.set_streamer_endpoint(streamer_endpoint)
    type_filter_node_pin.set_desired_type("speck2e::event::ReadoutPinValue")
    graph.start()

    return pin_buffer_node, power_buffer_node, graph, visualizer, gui_thread, power


def pin_readout(pin_buf) -> Tuple[int, bool]:
    recv_pin_events = pin_buf.get_events()
    # logging.info(f'--------------------------------------------------------recv_pin_events len: {len(recv_pin_events)}')

    # for ev in recv_pin_events:
    #     pred = ev.index
    #     time_stamp = ev.timestamp
    #     logging.info(f'ev.index: {ev.index}, ev.timestamp: {ev.timestamp}')

    valid = False
    if len(recv_pin_events):
        valid = True
        pin_event = recv_pin_events[-1]
        pred_label = pin_event.index - 1
        logging.info(f'valid: {valid}, pin pred label: {pred_label}, timestamp: {pin_event.timestamp}')
    else:
        pred_label = -1
        # logging.info(f'pin pred label: {pred_label}')

    return (pred_label, valid)


def get_power_consumption(power_buf, channels: List[str]) -> Dict[str, List[float]]:
    power_consumption = {chan: [] for chan in channels}

    events = power_buf.get_events()
    # logging.info(f'events_num: {events_num}')

    for ev in events:
        # logging.info(f'ev: type: {type(ev)}, {ev}, ev.timestamp: {ev.timestamp}')
        power_consumption[channels[ev.channel]].append(ev.value * 1000)  # to mW

    # for chan in channels:
    #     logging.info(f'{chan}, power: {power_per_channel[chan]}')

    return power_consumption


def concat_power_consumption(power_consumption: Dict[str, List[float]], one_sample_power_consumption: Dict[str, List[float]]):
    for channel in power_consumption.keys():
        power_consumption[channel].extend(one_sample_power_consumption[channel])


def write_power_consumption(channels: List[str], frequency: float, power_consumption: Dict[str, List[float]], save_file_path: str):
    time_interval = 1.0 / frequency

    with open(save_file_path, 'w') as fw:
        max_num = 0
        for value in power_consumption.values():
            if len(value) > max_num:
                max_num = len(value)

        times = [time_interval * inx for inx in range(max_num)]

        all_data = [times]
        for chan in channels:
            all_data.append(power_consumption[chan])

        fw.write(f'times, {", ".join(channels)}\n')
        for data in zip(*all_data):
            fw.write(f'{", ".join(map(str, data))}\n')


class PinValue(QThread):
    pred_result = Signal(int)

    def __init__(self):
        super(PinValue, self).__init__()

        config, devkit = model_config()

        # initialize samna
        self.time_interval = cfg.DYNAPCNN.TIME_INTERVAL
        pin_buf, power_buf, graph, visualizer, gui_thread, power = samna_initialization(devkit)

        self.pin_buf = pin_buf
        self.power_buf = power_buf
        self.graph = graph
        self.gui_thread = gui_thread
        self.power = power

        # power measurement
        self.frequency = cfg.POWER.FREQUENCY
        self.channels = cfg.POWER.CHANNELS
        self.write_pm_to_disk = cfg.POWER.WRITE_TO_DISK

        self.power.start_auto_power_measurement(self.frequency)
        time.sleep(3)

        if self.write_pm_to_disk:
            self.power_consumption = {chan: [] for chan in self.channels}
            self.save_file_path = os.path.join(os.path.dirname(cfg.MODEL.WEIGHTS), f'{cfg.POWER.SAVE_FILE_NAME}_{self.frequency}Hz.csv')

    def run(self):
        run_state_machine = False
        last_time = time.time()

        while 1:
            if self.isInterruptionRequested():
                self.power.stop_auto_power_measurement()
                time.sleep(2)

                self.gui_thread.join()
                self.graph.stop()

                if self.write_pm_to_disk:
                    write_power_consumption(self.channels, self.frequency, self.power_consumption, self.save_file_path)
                break

            pred, valid = pin_readout(self.pin_buf)
            if valid:
                run_state_machine = True
                last_time = time.time()
            else:
                if time.time() - last_time > self.time_interval:
                    run_state_machine = True
                    pred = -1
                    last_time = time.time()
                    logging.info(f'pin pred label: {pred}')
                else:
                    run_state_machine = False

            if run_state_machine:
                self.pred_result.emit(pred)
                run_state_machine = False

            if self.write_pm_to_disk:
                one_power_consumption = get_power_consumption(self.power_buf, self.channels)
                concat_power_consumption(self.power_consumption, one_power_consumption)


class ShowResult(QLabel):
    def __init__(self):
        super(ShowResult, self).__init__()

        self.pin_value = None
        self.pin_value = PinValue()
        self.pin_value.pred_result.connect(self.get_result)

        self.setFixedSize(480, 480)

        self.icons = read_icons()

        self.other_count = 0
        self.last_pred = None

        # state machine
        self.state_machine = FSM()

        self.pin_value.start()

    def closeEvent(self, event):
        if self.pin_value is not None:
            self.pin_value.requestInterruption()

            while self.pin_value.isRunning():
                time.sleep(0.5)

            self.pin_value = None

    @Slot(int)
    def get_result(self, pred: int):
        result = self.state_machine(pred)
        # logging.info(f'state machine output result: {result}')
        if result >= 0:
            logging.info(f'result: ---------------------------------------------{result}---- {os.path.basename(cfg.RESULT_IMGS[result])}')
            self.setPixmap(self.icons[result])
            self.last_result = result


def main():
    app = QApplication(sys.argv)

    show_result = ShowResult()

    show_result.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()