from threading import Timer
from threading import Lock
from logging import Logger
import numpy as np
from config import Config
from harvesters.core import Harvester

# --------------------
# GENICAM CAMERA ()
# --------------------
class CameraDevice:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.h = Harvester()
        # GenICam file
        self.h.add_file(Config.cti_path)
        self.h.update()
        self.logger.info('CameraDevice: GenICam %s', self.h.device_info_list)
        self.ia = None

        self._connected = False
        self._camera_state = 0
        self._sensor_type = 0
        self._camera_xsize = 640
        self._camera_ysize = 480
        self._binx = 1
        self._biny = 1
        self._bayeroffsetx = 0
        self._bayeroffsety = 0
        self._can_abort_exposure = False
        self._can_asymmetric_bin = False
        self._can_fast_readout = False
        self._can_get_cooler_power = False
        self._can_pulse_guide = False
        self._can_set_ccd_temperature = False
        self._can_stop_exposure = False
        self._ccd_temperature = 20
        self._cooler_on = False
        self._cooler_power = 0
        self._electrons_per_adu = 0
        self._exposure_max = 6e7 #us
        self._exposure_min = 20 #us
        self._exposure_resolution = 1
        self._fast_readout = 1.0
        self._full_well_capacity = 0
        self._gain = 0
        self._gain_max = 48
        self._gain_min = 0
        self._gains = [g for g in range(0, 48)]
        self._has_shutter = True
        self._heatsink_temperature = 0
        self._image_array = None
        self._image_array_variant = None
        self._imageready = False
        self._last_exposure_duration = 0
        self._last_exposure_start = 0
        self._max_adu = 255
        self._num_pixels = 192
        self._percent_completed = 0
        self._readout_mode = 0
        self._readout_modes = 0
        self._sensor_name = 'SVS'
        self._sensor_type = 0
        self._set_ccd_temperature = 20
        self._start_x = 0
        self._start_y = 0
        self._x_pixel_size = 5.94
        self._y_pixel_size = 5.94

        self._connecting = False
        self._num_y = 0
        self._num_x = 0
        logger.info('CameraDevice: _camera_ysize: %d', self._camera_ysize)
        logger.info('CameraDevice: _camera_xsize: %d', self._camera_xsize)
        

    def __str__(self):
        return f'CameraDevice: {CameraMetadata.Name}'
    
    def is_connected(self):
        return self._connected
    
    def connect(self):
        self._connecting = True
        self.logger.info('CameraDevice: try to connect')
        if self.ia is None:       
            self.ia = self.h.create()

            self.logger.info('CameraDevice: created ia')
            self.logger.info('CameraDevice: node_map:  %s', dir(self.ia.remote_device.node_map))
            # Sensor size
            self._camera_xsize = int(self.ia.remote_device.node_map.WidthMax.value)
            self._camera_ysize = int(self.ia.remote_device.node_map.HeightMax.value)
            self._num_x = self._camera_xsize
            self._num_y = self._camera_ysize
            self.logger.info('CameraDevice: _camera_ysize: %d', self._camera_ysize)
            self.logger.info('CameraDevice: _camera_xsize: %d', self._camera_xsize)


            # Set Pixel Format according to config.toml file
            self.ia.remote_device.node_map.PixelFormat.value = Config.PixelFormat

            # Freez Gain setting to Zero
            self.ia.remote_device.node_map.Gain.value = 0
            self.ia.remote_device.node_map.GainAuto.value = 'Off'
            self.ia.remote_device.node_map.GainAutoLevel.value = 0

            # set aquisition mode 
            self.ia.remote_device.node_map.AcquisitionMode.value = 'Continuous'  #'SingleFrame'

            # self.ia.start()     
            self._connected = True
            self._imageready = False

        self._connecting = False
        


    def disconnect(self):
        self._connected = False
        self._connecting = True
        if self.ia is not None:
            self.ia.stop()
            self.ia.destroy()
            self.ia = None
            self.h.reset()
        self._connecting = False

    def resetAndRestart(self):
        if self.ia is not None:
            self.ia.stop()
            self.ia.destroy()
            self.ia = self.h.create()
            self.ia.start()


    def set_exposure(self, exposure):
        # exposure from s to us
        self.logger.info('CameraDevice: set_exposure : %f s', exposure)
        exposure = int(exposure*1000000)
        self.ia.remote_device.node_map.ExposureTime.value = exposure
        self.logger.info('CameraDevice: Exposure set to: %s us', exposure)
    
    def get_exposure(self):
        exposure = self.ia.remote_device.node_map.ExposureTime.value
        # exposure from ms to s
        exposure = exposure/1000000
        self.logger.info('CameraDevice: Exposure: %s s', exposure)
        return exposure
    
    def set_gain(self, gain):
        if gain < self._gain_min:
            gain = self._gain_min
        if gain > self._gain_max:
            gain = self._gain_max

        self.ia.remote_device.node_map.Gain.value = gain
        self._gain = gain
        self.logger.info('CameraDevice: Gain set to: %s', gain)

    def capture_image(self):
        """
        Capture image from camera
        """
        self.logger.info('CameraDevice: capture_image')
        self.ia.start()
        
        # buffer = self.ia.fetch()
        # component = buffer.payload.components[0]
        # x2d = component.data.reshape(component.height, component.width)
        # buffer.queue()
        # frame = x2d.copy()

        with self.ia.fetch() as buffer:
                component = buffer.payload.components
                self.logger.info('CameraDevice: component size: %d', len(component))
                component = component[0]
                x2d = component.data.reshape(component.height, component.width)
                frame = x2d.copy()
        # converte to int32
        frame = frame.astype(np.int32)
        # transpose
        frame = np.transpose(frame)
        self.frame = frame
        self._imageready = True
        self.ia.stop()
        return frame

    def get_image(self):
        """
        retrive image from camera
        """
        self.logger.info('CameraDevice: get_image')  
        frame = self.frame        
        self.logger.info('CameraDevice: image shape: %s', frame.size)
        self.logger.info('CameraDevice: image type: %s', frame.dtype)
        self.logger.info('CameraDevice: image: %s', frame)
        
        self._imageready = False

        return frame
    
    def get_temperature(self):
        Temp = self.ia.remote_device.node_map.DeviceTemperature.value
        self.logger.info('CameraDevice: Temperature: %s', Temp)
        return Temp     

    
# if __name__ == '__main__':
#     import logging
#     logging.basicConfig(level=logging.DEBUG)
#     logger = logging.getLogger('CameraDeviceGENICAM')
#     cam = CameraDevice(logger)
#     cam.connect()
#     print(cam.get_image())
#     #cam.disconnect()
#     print(cam.is_connected())
