from ultralytics import YOLO

model = YOLO('best.pt')
results = model(source=0, conf=0.6,show=True,save=True)



