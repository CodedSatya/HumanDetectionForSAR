from ultralytics import YOLO

# model = YOLO('E:\\Projects\\College Projects\\HumanDetection\\runs\\detect\\train2\\weights\\last.pt')
model = YOLO('C:\\Projects\\College Projects\\HumanDetection\\HumanDetection\\model\\runs\\detect\\train5\\weights\\best.pt')


if __name__ == "__main__":
    # results = model.train(data='C:\Projects\College Projects\HumanDetection\HumanDetection\model\config.yaml', epochs=20, device=0)
    model.export(format="onnx")


