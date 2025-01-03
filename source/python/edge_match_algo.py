import time
import cv2 as cv
import numpy as np


class EdgeBaseMatch:
    def __init__(self, edge_threshold, nms_threshold):
        self.edge_threshold = edge_threshold
        self.nms_threshold = nms_threshold
        self.minScore = 0.8;
        self.greediness = 0.8;
        self.output = {}

    def run_match(self, target, img_xym, tpl_xym, pts):
        start = time.perf_counter()
        print("fuck match~~~~~~")
        # 基于候选点，全局匹配开始
        h, w = target.shape
        gx = img_xym[0]
        gy = img_xym[1]
        mag = img_xym[2]
        nMinScore = self.minScore / len(pts)
        nGreediness = (1 - self.greediness * self.minScore) / (1 - self.greediness) / len(pts)
        partialScore = 0
        row = 0
        min_step = 1
        boxes = []
        confidences = []
        while row < (h - 1):
            col = 0
            while col < (w - 1):
                sum = 0.0
                num = 0.0
                for pt in pts:
                    num += 1
                    curX = col + np.int(pt[0])
                    curY = row + np.int(pt[1])
                    if curX < 0 or curY < 0 or curX > w - 1 or curY > h - 1:
                        continue

                    # 目标边缘梯度
                    sdx = gx[curY, curX];
                    sdy = gy[curY, curX];

                    # 模板边缘梯度
                    tdx = tpl_xym[int(num - 1)][0];
                    tdy = tpl_xym[int(num - 1)][1]

                    # 计算匹配
                    if (sdx != 0 or sdy != 0) and (tdx != 0 or tdy != 0):
                        nMagnitude = mag[curY, curX]
                        if nMagnitude != 0:
                            sum += (sdx * tdx + sdy * tdy) * tpl_xym[int(num - 1)][2] / nMagnitude
                    partialScore = sum / num
                    if partialScore < min((self.minScore - 1) + (nGreediness * num), nMinScore * num)):
                        break
                if partialScore > self.edge_threshold:
                    results_pts = np.copy(pts)
                    for rspt in results_pts:
                        rspt[0] += col
                        rspt[1] += row
                    xx, yy, ww, hh = cv.boundingRect(results_pts)
                    boxes.append([xx, yy, ww, hh])
                    confidences.append(float(partialScore))
                col += min_step
            row += min_step

        nms_indices = cv.dnn.NMSBoxes(boxes, confidences, self.edge_threshold, self.nms_threshold)
        det_boxes = []
        scores = []
        for i in range(len(nms_indices)):
            rect_box = boxes[nms_indices[i]]
            det_boxes.append(rect_box)
            scores.append(confidences[nms_indices[i]])
        self.output['boxes'] = det_boxes
        self.output['scores'] = scores
        end = time.perf_counter()
        print(f'Finished in {round(end - start, 4)} seconds')
        return self.output


def xy_gradient(image):
    gx = cv.Sobel(image, cv.CV_32F, 1, 0)
    gy = cv.Sobel(image, cv.CV_32F, 0, 1)
    magnitude, direction = cv.cartToPolar(gx, gy)
    return gx, gy, magnitude, direction


if __name__ == "__main__":
    print("test ncc......")
    tpl_image = cv.imread("D:/images/vm_test/tpl_01.png")
    target_image = cv.imread("D:/images/vm_test/target2.jpg")


    tpl_gray = cv.cvtColor(tpl_image, cv.COLOR_BGR2GRAY)
    target_gray = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)
    print("run once~~~~")
    # 梯度提取
    t_gx, t_gy, t_mag, t_dire = xy_gradient(tpl_gray)
    gx, gy, mag, dire = xy_gradient(target_gray)
    th, tw = tpl_gray.shape
    h, w = target_gray.shape

    # 提取候选点
    binary = cv.adaptiveThreshold(tpl_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 10)
    cv.imwrite("D:/tttt.png", binary)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    index = 0
    max = 0
    for c in range(len(contours)):
        area = cv.contourArea(contours[c])
        if area > max:
            max = area
            index = c
    pts = np.array(contours[index]).reshape((-1, 2))
    # 提取相对位置信息
    original_x = pts[0][0]
    original_y = pts[0][1]
    t_xym = []
    for p in pts:
        x = np.int(p[0])
        y = np.int(p[1])
        p[0] = x - original_x
        p[1] = y - original_y
        if t_mag[y, x] == 0.0:
            t_xym.append([t_gx[y, x], t_gy[y, x], 0.0])
        else:
            t_xym.append([t_gx[y, x], t_gy[y, x], 1.0 / t_mag[y, x]])

    edge_match = EdgeBaseMatch(0.8, 0.5)
    output = edge_match.run_match(target_gray, [gx, gy, mag], t_xym, pts)
    det_boxes = output['boxes']
    for rect_box in det_boxes:
        cv.rectangle(target_image, (rect_box[0], rect_box[1]),
                     (rect_box[0] + rect_box[2], rect_box[1] + rect_box[3]), (0, 0, 255), 2, 8, 0)
    cv.imshow("result", target_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
