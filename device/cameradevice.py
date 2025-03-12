from threading import Timer
from threading import Lock
from logging import Logger
import numpy as np
import cv2
# --------------------
# SIMULATED CAMERA ()
# --------------------
class CameraDevice:
    def __init__(self, logger: Logger):
        self.logger = logger
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
        self._exposure_max = 10.0
        self._exposure_min = 0.00001
        self._exposure_resolution = 1e-6
        self._fast_readout = 1.0
        self._full_well_capacity = 0
        self._gain = 1.0
        self._gain_max = 30.
        self._gain_min = 0.0
        self._gains = 1.0
        self._has_shutter = True
        self._heatsink_temperature = 0
        self._image_array = None
        self._image_array_variant = None
        self._imageready = False
        self._last_exposure_duration = 0
        self._last_exposure_start = 0
        self._max_adu = 65535
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
        #self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
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
        self.logger.info('CameraDevice: connect')
        self._connecting = True
        # use directshow
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self._camera_xsize = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._camera_ysize = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._num_x = self._camera_xsize
        self._num_y = self._camera_ysize
        self.logger.info('CameraDevice: _camera_ysize: %d', self._camera_ysize)
        self.logger.info('CameraDevice: _camera_xsize: %d', self._camera_xsize)
        self._connecting = False
        self._connected = True


    def disconnect(self):
        self._connected = False
        if self.cam is not None:
            self._connecting = True
            self.cam.release()
            self._connecting = False

    def get_image(self):
        self._imageready = False
        ret, frame = self.cam.read()
        #convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #coverte to int32
        frame = frame.astype(np.int32)
        # transpose
        frame = np.transpose(frame)       

        #frame = np.random.randint(0, 65535, (self._camera_xsize, self._camera_ysize), dtype=np.int32)
        self.logger.info('CameraDevice: image shape: %s', frame.size)
        self.logger.info('CameraDevice: image type: %s', frame.dtype)
        self.logger.info('CameraDevice: image: %s', frame)

        
        self._imageready = True
        return frame
    
    def get_temperature(self):
        self._ccd_temperature += np.random.uniform(-2., 2.)
        return self._ccd_temperature
    
# if __name__ == '__main__':
#     import logging
#     logging.basicConfig(level=logging.DEBUG)
#     logger = logging.getLogger('CameraDevice')
#     cam = CameraDevice(logger)
#     cam.connect()
#     print(cam.get_image())
#     #cam.disconnect()
#     print(cam.is_connected())
