import cv2 as cv
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class YOLOv5TensorRTDetector:

    def __init__(self, settings):
        self.cfx = cuda.Device(0).make_context()
        self.input_w = 640
        self.input_h = 640
        self.infer_settings = settings
        self.resnet_model = None
        self.means = np.zeros((224, 224, 3), dtype=np.float32)
        self.means[:, :] = (0.485, 0.456, 0.406)
        self.dev = np.zeros((224, 224, 3), dtype=np.float32)
        self.dev[:, :] = (0.229, 0.224, 0.225)
        with open(self.infer_settings.label_map_file_path) as f:
            self.labels = [line.strip() for line in f.readlines()]
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.infer_settings.weight_file_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            if self.engine.binding_is_input(index):
                print("input name: ", name)
            else:
                print("output name: ", name)
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            shape = self.engine.get_binding_shape(index)
            data = np.zeros(shape, dtype=np.float32)
            print(shape)
            allocation = cuda.mem_alloc(data[0].nbytes)
            binding = {
                "index": index,
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(index):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def output_spec(self):
        specs = []
        for o in self.outputs:
            specs.append((o["shape"], o["dtype"]))
        return specs

    def format_yolov5(self, frame):
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    def post_process(self, input_image, output_data):
        class_ids = []
        confidences = []
        boxes = []
        rows = output_data.shape[0]

        image_width, image_height, _ = input_image.shape

        x_factor = image_width / self.input_w
        y_factor = image_height / self.input_h

        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= self.infer_settings.conf_threshold:

                classes_scores = row[5:]
                _, _, _, max_indx = cv.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > self.infer_settings.score_threshold):
                    confidences.append(confidence)

                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indexes = cv.dnn.NMSBoxes(boxes, confidences, self.infer_settings.score_threshold, self.infer_settings.nms_threshold)

        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])
        return result_class_ids, result_confidences, result_boxes

    def infer_image(self, frame):
        # 此处必须添加push
        start = time.time()
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        inputImage = self.format_yolov5(rgb)
        input_image = cv.resize(src=inputImage, dsize=(self.input_w, self.input_h))
        blob_img = np.float32(input_image) / 255.0
        input_x = blob_img.transpose((2, 0, 1))
        input_blob = np.expand_dims(input_x, 0)

        self.cfx.push()
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))
        cuda.memcpy_htod(self.inputs[0]["allocation"], np.ascontiguousarray(input_blob))
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]["allocation"])
        self.cfx.pop()

        class_ids, confidences, boxes = self.post_process(inputImage, outputs[3][0])
        if self.infer_settings.show_fps:
            end = time.time()
            inf_end = end - start
            fps = 1 / inf_end
            fps_label = "FPS: %.2f" % fps
            cv.putText(frame, fps_label, (20, 45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            cv.rectangle(frame, box, (140, 199, 0), 2)
            cv.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (255, 255, 255), -1)
            cv.putText(frame, "{0}: {1:.2f}".format(self.labels[classid], confidence), (box[0], box[1] - 5),
                       cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

    def __del__(self):
        self.cfx.pop()
        del self.cfx


if __name__ == "__main__":
    from dlcore.dl_infer_settings import DLInferSettings
    settings = DLInferSettings()
    settings.weight_file_path = "D:/projects/yolov5s.engine"
    settings.label_map_file_path = "D:/projects/classes.txt"
    classifier = YOLOv5TensorRTDetector(settings)
    image = cv.imread("D:/images/horses.jpg")
    classifier.infer_image(image)
    cv.imshow("result", image)
    cv.waitKey(0)
    cv.destroyAllWindows()