import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

import tensorrt as trt
import torchvision.transforms as transforms

import cv2


import PIL.Image as pil
import matplotlib as mpl
import numpy

import os
import glob


import matplotlib.cm as cm

import time

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()


def gstreamer_pipeline(
    capture_width=640,
    capture_height=192,
    display_width=640,
    display_height=192,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()

        builder.max_batch_size = 1
        config.max_workspace_size = 1 << 30

        # allow TensorRT to use up to 1GB of GPU memory for tactic selection
        #builder.max_workspace_size = 1 << 30
        # we have only one image in batch
        #builder.max_batch_size = 1
        # use FP16 mode if possible

        #builder.fp16_mode = True
        #if builder.platform_has_fast_int8:
        #    builder.int8_mode = True
        #    builder.int8_calibrator = Int8_calibrator
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:

            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        #last_layer = network.get_layer(network.num_layers-1)
        #network.mark_output(last_layer.get_output(0))
        print('Completed parsing of ONNX file')

        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        plan = builder.build_serialized_network(network, config)
        with trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(plan)
        #engine = builder.build_cuda_engine(network)
        with open("./model.engine", "wb") as f:
            f.write(engine.serialize())
        print("Completed creating Engine")

        # generate TensorRT engine optimized for the target platform
        print('Building an engine...')
        #engine = builder.build_cuda_engine(network)
        context = engine.create_execution_context()
        print("Completed creating Engine")


        with open("model.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()
        return engine, context


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def _preprocess_trt(img, shape=(640, 192)):
    """Preprocess an image before TRT SSD inferencing."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, shape)
    img = img.astype(np.float32)
    return img

class Detector():

    def _load_engine(self):
        with open("model.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def _create_context(self):
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return self.engine.create_execution_context()


    def __init__(self):
        self.colors = np.random.uniform(0, 255, size=(100, 3))
        self.input_shape = (640, 192)
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.context = self._create_context()

    def __del__(self):
        """Free CUDA memories."""
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs



    def prediction(self, img):

        img_resized = img
        np.copyto(self.host_inputs[0], img_resized.ravel())
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        output = self.host_outputs[0]
        return output


if __name__ == "__main__":

    
    #first run for build engine
    engine, context = build_engine("./full_mobilenet_indoor.onnx")
    
    detector = Detector()
    #cap = cv2.VideoCapture('test.mp4') #input video
   
    cap = cv2.VideoCapture(0)  # says we capture an image from a webcam
    if (cap.isOpened()== False):
        print("can not open camera")
   
    while (cap.isOpened()):
     
        ret, frame = cap.read()

        frame = cv2.resize(frame, (640, 480))
        
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow('Frame1', frame)
            
        input_image = pil.fromarray(cv2_im)
        
        input_image = transforms.ToTensor()(input_image).unsqueeze(0).numpy()

        output = detector.prediction(input_image)
        output = output.reshape(1, 1, 240, 320)
           
        disp_resized = output

        scaled_disp, _ = disp_to_depth(output, 0.1, 60)
        #print(scaled_disp)
        
        disp_resized_np = disp_resized.squeeze()
        #print(disp_resized_np.shape)
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        open_cv_image = numpy.array(im)
        
        open_cv_image = cv2.resize(open_cv_image, (720, 720))
        cv2.imshow('Depth map', open_cv_image)
        
         
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            




