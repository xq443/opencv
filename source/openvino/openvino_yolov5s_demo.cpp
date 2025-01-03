#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace InferenceEngine;
string model_xml = "D:/projects/models/face-detection-0102/FP32/face-detection-0102.xml";
string model_bin = "D:/projects/models/face-detection-0102/FP32/face-detection-0102.bin";

std::vector<float> anchors = {
	10,13, 16,30, 33,23,
	30,61, 62,45, 59,119,
	116,90, 156,198, 373,326
};

int get_anchor_index(int scale_w, int scale_h) {
	if (scale_w == 20) {
		return 12;
	}
	if (scale_w == 40) {
		return 6;
	}
	if (scale_w == 80) {
		return 0;
	}
	return -1;
}

float get_stride(int scale_w, int scale_h) {
	if (scale_w == 20) {
		return 32.0;
	}
	if (scale_w == 40) {
		return 16.0;
	}
	if (scale_w == 80) {
		return 8.0;
	}
	return -1;
}

float sigmoid_function(float a)
{
	float b = 1. / (1. + exp(-a));
	return b;
}

void face_detection_demo();
void yolov5_onnx_demo();
int main(int argc, char** argv) {
	yolov5_onnx_demo();
}

void yolov5_onnx_demo() {
	Mat src = imread("D:/python/yolov5/data/images/zidane.jpg");
	int image_height = src.rows;
	int image_width = src.cols;

	// 创建IE插件, 查询支持硬件设备
	Core ie;
	vector<string> availableDevices = ie.GetAvailableDevices();
	for (int i = 0; i < availableDevices.size(); i++) {
		printf("supported device name : %s \n", availableDevices[i].c_str());
	}

	//  加载检测模型
	auto network = ie.ReadNetwork("D:/python/yolov5/yolov5s.xml", "D:/python/yolov5/yolov5s.bin");
	// auto network = ie.ReadNetwork("D:/python/yolov5/yolov5s.onnx");

	// 请求网络输入与输出信息
	InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
	InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());
	// 设置输入格式
	for (auto &item : input_info) {
		auto input_data = item.second;
		input_data->setPrecision(Precision::FP32);
		input_data->setLayout(Layout::NCHW);
		input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
		input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
	}

	// 设置输出格式
	for (auto &item : output_info) {
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
	}
	auto executable_network = ie.LoadNetwork(network, "GPU");

	// 处理解析输出结果
	vector<Rect> boxes;
	vector<int> classIds;
	vector<float> confidences;

	// 请求推断图
	auto infer_request = executable_network.CreateInferRequest();
	float scale_x = image_width / 640.0;
	float scale_y = image_height / 640.0;

	int64 start = getTickCount();
	/** Iterating over all input blobs **/
	for (auto & item : input_info) {
		auto input_name = item.first;

		/** Getting input blob **/
		auto input = infer_request.GetBlob(input_name);
		size_t num_channels = input->getTensorDesc().getDims()[1];
		size_t h = input->getTensorDesc().getDims()[2];
		size_t w = input->getTensorDesc().getDims()[3];
		size_t image_size = h*w;
		Mat blob_image;
		resize(src, blob_image, Size(w, h));
		cvtColor(blob_image, blob_image, COLOR_BGR2RGB);

		// NCHW
		float* data = static_cast<float*>(input->buffer());
		for (size_t row = 0; row < h; row++) {
			for (size_t col = 0; col < w; col++) {
				for (size_t ch = 0; ch < num_channels; ch++) {
					data[image_size*ch + row*w + col] = float(blob_image.at<Vec3b>(row, col)[ch])/255.0;
				}
			}
		}
	}

	// 执行预测
	infer_request.Infer();

	for (auto &item : output_info) {
		auto output_name = item.first;
		printf("output_name : %s \n", output_name.c_str());
		// 获取输出数据
		auto output = infer_request.GetBlob(output_name);

		const float* output_blob = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
		const SizeVector outputDims = output->getTensorDesc().getDims();
		const int out_n = outputDims[0];
		const int out_c = outputDims[1];
		const int side_h = outputDims[2];
		const int side_w = outputDims[3];
		const int side_data = outputDims[4];
		float stride = get_stride(side_h, side_h);
		int anchor_index = get_anchor_index(side_h, side_h);
		printf("number of images: %d, channels : %d, height: %d, width : %d, out_data:%d \n", out_n, out_c, side_h, side_w, side_data);
		int side_square = side_h*side_w;
		int side_data_square = side_square*side_data;
		int side_data_w = side_w*side_data;
		for (int i = 0; i < side_square; ++i) {
			for (int c = 0; c < out_c; c++) {
				int row = i / side_h;
				int col = i % side_h;
				int object_index = c*side_data_square + row*side_data_w + col*side_data;

				// 阈值过滤
				float conf = sigmoid_function(output_blob[object_index + 4]);
				if (conf < 0.25) {
					continue;
				}

				// 解析cx, cy, width, height
				float x = (sigmoid_function(output_blob[object_index]) * 2 - 0.5 + col)*stride;
				float y = (sigmoid_function(output_blob[object_index + 1]) * 2 - 0.5 + row)*stride;
				float w = pow(sigmoid_function(output_blob[object_index + 2]) * 2, 2)*anchors[anchor_index + c * 2];
				float h = pow(sigmoid_function(output_blob[object_index + 3]) * 2, 2)*anchors[anchor_index + c * 2 + 1];
				float max_prob = -1;
				int class_index = -1;

				// 解析类别
				for (int d = 5; d < 85; d++) {
					float prob = sigmoid_function(output_blob[object_index + d]);
					if (prob > max_prob) {
						max_prob = prob;
						class_index = d - 5;
					}
				}

				// 转换为top-left, bottom-right坐标
				int x1 = saturate_cast<int>((x - w / 2) * scale_x);  // top left x
				int y1 = saturate_cast<int>((y - h / 2) * scale_y);  // top left y
				int x2 = saturate_cast<int>((x + w / 2) * scale_x);  // bottom right x
				int y2 = saturate_cast<int>((y + h / 2) * scale_y); // bottom right y

				// 解析输出
				classIds.push_back(class_index);
				confidences.push_back((float)conf);
				boxes.push_back(Rect(x1, y1, x2 - x1, y2 - y1));
				// rectangle(src, Rect(x1, y1, x2 - x1, y2 - y1), Scalar(255, 0, 255), 2, 8, 0);
			}
		}
	}

	vector<int> indices;
	NMSBoxes(boxes, confidences, 0.25, 0.5, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		rectangle(src, box, Scalar(140, 199, 0), 4, 8, 0);
	}
	float fps = getTickFrequency() / (getTickCount() - start);
	float time = (getTickCount() - start) / getTickFrequency();

	ostringstream ss;
	ss << "FPS : " << fps << " detection time: " << time * 1000 << " ms";
	putText(src, ss.str(), Point(20, 50), 0, 1.0, Scalar(0, 0, 255), 2);

	imshow("OpenVINO2021R2+YOLOv5对象检测", src);
	imwrite("D:/openvino2021_yolov5_test.png", src);
	waitKey(0);
}

void face_detection_demo() {
	Mat src = imread("D:/images/persons.png");
	int image_height = src.rows;
	int image_width = src.cols;

	// 创建IE插件, 查询支持硬件设备
	Core ie;
	vector<string> availableDevices = ie.GetAvailableDevices();
	for (int i = 0; i < availableDevices.size(); i++) {
		printf("supported device name : %s \n", availableDevices[i].c_str());
	}

	//  加载检测模型
	auto network = ie.ReadNetwork(model_xml, model_bin);

	// 请求网络输入与输出信息
	InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
	InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());
	// 设置输入格式
	for (auto &item : input_info) {
		auto input_data = item.second;
		input_data->setPrecision(Precision::U8);
		input_data->setLayout(Layout::NCHW);
		input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
		input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
	}
	printf("get it \n");

	// 设置输出格式
	for (auto &item : output_info) {
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
	}

	// 创建可执行网络对象
	// ie.AddExtension(std::make_shared<Extension>("C:/Intel/openvino_2020.1.033/deployment_tools/ngraph/lib/ngraph.dll"), "CPU");
	auto executable_network = ie.LoadNetwork(network, "CPU");
	// auto executable_network = ie.LoadNetwork(network, "MYRIAD");

	// 请求推断图
	auto infer_request = executable_network.CreateInferRequest();

	/** Iterating over all input blobs **/
	for (auto & item : input_info) {
		auto input_name = item.first;

		/** Getting input blob **/
		auto input = infer_request.GetBlob(input_name);
		size_t num_channels = input->getTensorDesc().getDims()[1];
		size_t h = input->getTensorDesc().getDims()[2];
		size_t w = input->getTensorDesc().getDims()[3];
		size_t image_size = h*w;
		Mat blob_image;
		resize(src, blob_image, Size(w, h));

		// NCHW
		unsigned char* data = static_cast<unsigned char*>(input->buffer());
		for (size_t row = 0; row < h; row++) {
			for (size_t col = 0; col < w; col++) {
				for (size_t ch = 0; ch < num_channels; ch++) {
					data[image_size*ch + row*w + col] = blob_image.at<Vec3b>(row, col)[ch];
				}
			}
		}
	}

	// 执行预测
	infer_request.Infer();

	// 处理输出结果
	for (auto &item : output_info) {
		auto output_name = item.first;

		// 获取输出数据
		auto output = infer_request.GetBlob(output_name);
		const float* detection = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
		const SizeVector outputDims = output->getTensorDesc().getDims();
		const int maxProposalCount = outputDims[2];
		const int objectSize = outputDims[3];

		// 解析输出结果
		for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
			float label = detection[curProposal * objectSize + 1];
			float confidence = detection[curProposal * objectSize + 2];
			float xmin = detection[curProposal * objectSize + 3] * image_width;
			float ymin = detection[curProposal * objectSize + 4] * image_height;
			float xmax = detection[curProposal * objectSize + 5] * image_width;
			float ymax = detection[curProposal * objectSize + 6] * image_height;
			if (confidence > 0.5) {
				printf("label id : %d\n", static_cast<int>(label));
				Rect rect;
				rect.x = static_cast<int>(xmin);
				rect.y = static_cast<int>(ymin);
				rect.width = static_cast<int>(xmax - xmin);
				rect.height = static_cast<int>(ymax - ymin);
				putText(src, "OpenVINO-2021R02", Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2, 8);
				rectangle(src, rect, Scalar(0, 255, 255), 2, 8, 0);
			}
			std::cout << std::endl;
		}
	}
	imshow("openvino+ssd人脸检测", src);
	imwrite("D:/result.png", src);
	waitKey(0);
	return;
}
