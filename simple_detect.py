import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import check_img_size, increment_path, non_max_suppression, check_imshow, cv2, scale_coords, xyxy2xywh
from utils.dataloaders import LoadImages, LoadStreams
from utils.torch_utils import time_sync
from utils.plots import Annotator, colors
import torch.backends.cudnn as cudnn

weights = "Best.pt"
source = "0"
imgsz = (640, 640)
bs = 1
conf_thres = 0.6
iou_thres = 0.45
classes = None
agnostic_nms = False
max_det = 1000
webcam = True 
line_thickness = 3

save_dir = increment_path(Path("runs/detect") / "exp", exist_ok=False)
model = DetectMultiBackend(weights)
stride, names, pt = model.stride, model.names, model.pt

imgsz = check_img_size(imgsz, s=stride)
view_img = check_imshow()
cudnn.benchmark = True  # set True to speed up constant image size inference
dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
bs = len(dataset)  # batch_size
vid_path, vid_writer = [None] * bs, [None] * bs
#dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
seen, windows, dt = 0, [], [0.0, 0.0, 0.0]

for path, im, im0s, vid_cap, s in dataset:
  t1 = time_sync()
  im = torch.from_numpy(im).to("cpu")
  im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
  im /= 255
  if len(im.shape) == 3:
    im = im[None]  # expand for batch dim
  t2 = time_sync()
  dt[0] += t2 - t1
  t3 = time_sync()
  dt[1] += t3 - t2

  pred = model(im)
  pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
  dt[2] += time_sync() - t3
  

  # Process predictions
  for i, det in enumerate(pred):  # per image
    seen += 1
    if webcam:  # batch_size >= 1
        p, im0, frame = path[i], im0s[i].copy(), dataset.count
        s += f'{i}: '
    p = Path(p)
    s += '%gx%g ' % im.shape[2:]  # print string
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
    if len(det):
      # Rescale boxes from img_size to im0 size
      det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

      # Print results
      for c in det[:, -1].unique():
          n = (det[:, -1] == c).sum()  # detections per class
          s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

      # Write results
      for *xyxy, conf, cls in reversed(det):
          if view_img:  # Add bbox to image
              c = int(cls)  # integer class
              label = f'{names[c]} {conf:.2f}'
              annotator.box_label(xyxy, label, color=colors(c, True))
    im0 = annotator.result()
    if view_img:
        if p not in windows:
            windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)  # 1 millisecond