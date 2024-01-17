import json

from flask import Flask, request
from flask_cors import cross_origin
from ultralytics import YOLO

app = Flask(__name__)

@app.route('/api', methods=['POST'])
@cross_origin()
def api():
    # 在这里处理POST请求的逻辑

    # 获取POST请求的数据
    data = request.values.get("picPath")
    print(data)

    # 加载预训练模型
    model = YOLO("/Users/lixizi/Documents/project/project_file/server/best.pt", task='detect')
    # model = YOLO("yolov8n.pt") task参数也可以不填写，它会根据模型去识别相应任务类别
    # 检测图片
    # results = model("20230926132112.jpg")

    results = model(data)
    result = results[0]
    resultlist = []
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)

        resultlist.append({"label": class_id, "x1": cords[0], "y1": cords[1], "x2": cords[2], "y2": cords[3], "prob": conf})

    jsonstr = json.dumps(resultlist)
    print(jsonstr)

    return jsonstr

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)
    # app.run(host="192.168.43.14", port=8000)