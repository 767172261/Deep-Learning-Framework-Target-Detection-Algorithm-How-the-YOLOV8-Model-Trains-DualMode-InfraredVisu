深度学习框架目标检测算法YOLOV8模型如何训练 红外-可见光对齐目标检测数据集  红外目标检测数据集的训练及应用
@[toc]

**数据集描述：**

红外目标检测数据集。红外-可见光对齐目标检测数据集。总共4200个图片对，共8400张。voc、yolo标注格式
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dfaffeb29e974ce289eaa27d417e5b78.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2183fe96627a4b8a84a85497974f4f4c.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/68d51d1b46a546319b247a4005f7413b.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4ecdff92a9b4438797ccc15c79b91540.png)
使用 **YOLOv8** 实现双模态（红外 + 可见光）目标检测系统的构建。

> ✅ **目标：**
> - 使用 **红外和可见光图像对** 进行目标检测
> - 支持 VOC / YOLO 标注格式
> - 使用官方预训练 YOLOv8 模型进行迁移学习
> - 提供完整训练、推理、评估代码
> - 包含 CUDA、Anaconda、虚拟环境、依赖安装等全过程

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/29e2fdeff27e43e2a0b8b4ade9cf32e1.png)

## 🛠️ 一、环境搭建全流程（Windows/Linux）

### 1. 安装 CUDA 驱动（GPU 支持）

#### ✅ 检查显卡驱动是否支持 CUDA
```bash
nvidia-smi
```
输出应显示驱动版本和 CUDA 版本（如 `CUDA Version: 12.2`）。

> ⚠️ 若无输出，请安装最新 NVIDIA 驱动：[https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)

#### ✅ 安装 PyTorch 所需的 CUDA（通过 PyTorch 官网）
访问：[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

选择命令（推荐 CUDA 11.8 或 12.1）：
```bash
# 示例：CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

验证：
```python
import torch
print(torch.cuda.is_available())        # 应为 True
print(torch.__version__)                # 查看版本
```

---

### 2. 安装 Anaconda

下载地址：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

安装后验证：
```bash
conda --version
```

---

### 3. 创建 Python 虚拟环境

```bash
# 创建环境（Python 3.9 推荐）
conda create -n infrared_detection python=3.9
conda activate infrared_detection

# 升级 pip
pip install --upgrade pip
```

---

### 4. 安装必要依赖

```bash
pip install ultralytics opencv-python matplotlib numpy tqdm pillow scikit-learn
```

> ✅ `ultralytics`：YOLOv8 官方库  
> ✅ `opencv-python`：图像处理  
> ✅ `matplotlib`：可视化  
> ✅ `tqdm`：进度条

验证安装：
```python
from ultralytics import YOLO
print("✅ Ultralytics installed!")
```

---

## 📁 二、数据集结构设计（红外-可见光对齐）



### ✅ 方案 A：双通道输入（推荐用于对齐数据）
将红外与可见光拼接为 **3通道输入（如 R=IR, G=VIS, B=IR）** 或 **4通道（需修改模型）**

### ✅ 方案 B：分别训练（简单易行）
- 分别在红外图像上训练一个模型
- 在可见光图像上训练另一个模型
- 后期融合结果（本教程采用此方式演示）

---

### 数据集结构（YOLO 格式）

```
infrared_visible_dataset/
├── images/
│   ├── train/
│   │   ├── ir_0001.jpg
│   │   ├── vis_0001.jpg
│   │   └── ...
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── ir_0001.txt
│   │   ├── vis_0001.txt
│   │   └── ...
│   ├── val/
│   └── test/
├── data_ir.yaml
├── data_vis.yaml
```

### `data_ir.yaml`（红外数据）

```yaml
train: ./infrared_visible_dataset/images/train_ir
val: ./infrared_visible_dataset/images/val_ir
test: ./infrared_visible_dataset/images/test_ir

nc: 4
names: ['Plane', 'Drone', 'Heli', 'Bird']
```

### `data_vis.yaml`（可见光数据）

```yaml
train: ./infrared_visible_dataset/images/train_vis
val: ./infrared_visible_dataset/images/val_vis
test: ./infrared_visible_dataset/images/test_vis

nc: 4
names: ['Plane', 'Drone', 'Heli', 'Bird']
```

> 💡 说明：
> - `train_ir`, `val_ir`：只包含红外图像
> - `train_vis`, `val_vis`：只包含可见光图像
> - `.txt` 文件格式为 YOLO：`class_id x_center y_center width height`（归一化）

---

## 🚀 三、调用官方预训练模型并训练（YOLOv8）

### 1. 训练红外图像模型（train_ir.py）

```python
# train_ir.py
from ultralytics import YOLO
import torch

def train_infrared_model():
    # 检查设备
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载官方预训练模型（推荐 yolov8s 或 yolov8m）
    model = YOLO('yolov8s.pt')  # 自动下载

    # 开始训练
    results = model.train(
        data='infrared_visible_dataset/data_ir.yaml',
        epochs=150,
        imgsz=640,
        batch=16,
        name='ir_yolov8s',
        project='runs/detect',
        save=True,
        save_period=10,
        device=device,
        workers=4,
        patience=30,
        optimizer='AdamW',
        lr0=0.001,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.2,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1
    )

    print("✅ 红外模型训练完成！")
    return results

if __name__ == '__main__':
    train_infrared_model()
```

### 2. 训练可见光模型（train_vis.py）

```python
# train_vis.py
from ultralytics import YOLO
import torch

def train_visible_model():
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8s.pt')

    results = model.train(
        data='infrared_visible_dataset/data_vis.yaml',
        epochs=150,
        imgsz=640,
        batch=16,
        name='vis_yolov8s',
        project='runs/detect',
        save=True,
        device=device,
        workers=4,
        patience=30,
        lr0=0.001
    )

    print("✅ 可见光模型训练完成！")
    return results

if __name__ == '__main__':
    train_visible_model()
```

> 📦 模型保存路径：
> - 红外：`runs/detect/ir_yolov8s/weights/best.pt`
> - 可见光：`runs/detect/vis_yolov8s/weights/best.pt`

---

## 🔍 四、推理代码（detect.py）

### 单图推理（支持双模态）

```python
# detect.py
from ultralytics import YOLO
import cv2
import numpy as np

class MultiModalDetector:
    def __init__(self, ir_model_path='runs/detect/ir_yolov8s/weights/best.pt',
                       vis_model_path='runs/detect/vis_yolov8s/weights/best.pt'):
        self.ir_model = YOLO(ir_model_path)
        self.vis_model = YOLO(vis_model_path)

    def detect(self, ir_image_path, vis_image_path, conf=0.4, output_path='fusion_result.jpg'):
        # 红外检测
        ir_results = self.ir_model.predict(source=ir_image_path, conf=conf, verbose=False)
        ir_img = ir_results[0].plot()

        # 可见光检测
        vis_results = self.vis_model.predict(source=vis_image_path, conf=conf, verbose=False)
        vis_img = vis_results[0].plot()

        # 拼接对比图
        combined = np.hstack([ir_img, vis_img])
        cv2.imwrite(output_path, combined)
        print(f"✅ 检测完成，结果保存至: {output_path}")

        # 返回原始检测结果（可用于融合）
        return ir_results[0], vis_results[0]

# 使用示例
if __name__ == '__main__':
    detector = MultiModalDetector()
    ir_result, vis_result = detector.detect(
        ir_image_path='test_ir.jpg',
        vis_image_path='test_vis.jpg'
    )
```

---

## 📊 五、评估代码（evaluate.py）

```python
# evaluate.py
from ultralytics import YOLO

def evaluate_model(model_path, data_yaml):
    model = YOLO(model_path)
    
    metrics = model.val(
        data=data_yaml,
        batch=16,
        imgsz=640,
        device=0 if torch.cuda.is_available() else 'cpu',
        save_json=True,
        conf_thres=0.001,
        iou_thres=0.6
    )

    print(f"✅ mAP@0.5: {metrics.box.map50:.4f}")
    print(f"✅ mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p:.4f}")
    print(f"Recall: {metrics.box.r:.4f}")

    print("\n📊 各类别 AP@0.5:")
    for i, name in enumerate(metrics.names.values()):
        print(f"  {name}: {metrics.box.ap[i]:.4f}")

# 评估红外模型
print("🔍 红外模型评估:")
evaluate_model('runs/detect/ir_yolov8s/weights/best.pt', 'infrared_visible_dataset/data_ir.yaml')

# 评估可见光模型
print("\n🔍 可见光模型评估:")
evaluate_model('runs/detect/vis_yolov8s/weights/best.pt', 'infrared_visible_dataset/data_vis.yaml')
```

---

## 🧠 六、高级功能建议（可选）

### ✅ 1. 多模态融合策略（后期融合）
```python
def fuse_results(ir_result, vis_result, iou_thres=0.5):
    from ultralytics.engine.results import Results
    # 使用 NMS 或 Soft-NMS 融合两个检测结果
    # 可基于置信度加权平均
    pass
```

### ✅ 2. 导出为 ONNX/TensorRT（加速部署）
```python
model = YOLO('runs/detect/ir_yolov8s/weights/best.pt')
model.export(format='onnx', dynamic=True, opset=13)
model.export(format='engine', device=0)  # TensorRT
```

### ✅ 3. 构建 Web 系统（Flask）
```python
from flask import Flask, request, send_file
app = Flask(__name__)
detector = MultiModalDetector()

@app.route('/detect', methods=['POST'])
def run_detection():
    ir = request.files['ir']
    vis = request.files['vis']
    ir_path = 'temp_ir.jpg'
    vis_path = 'temp_vis.jpg'
    ir.save(ir_path)
    vis.save(vis_path)
    detector.detect(ir_path, vis_path, output_path='result.jpg')
    return send_file('result.jpg', mimetype='image/jpeg')
```

---

以上文字及代码仅供参考学习使用。
