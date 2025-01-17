from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2



def object_count_test():
    # 权重文件，可替换为自己训练的权重文件
    model = YOLO("runs/detect/train50/weights/best.pt")
    # results = model.train(data='VisDrone.yaml', epochs=3)
    # 实际场景为视频流地址
    cap = cv2.VideoCapture("ultralytics_rc_output/exp2/VID_20240223_152846.mp4")
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define region points
    # 视频中我们检测线的大小、宽细以及横线在视频中的横纵坐标
    region_points = [(0, 0), (0, 0), (0, 0), (0, 0)]
    # person and car classes for count
    # classes_to_count = [0, 2]

    # Video writer
    video_writer = cv2.VideoWriter("object_counting_output.avi",
                           cv2.VideoWriter_fourcc(*'mp4v'),
                           fps,
                           (w, h))

    # Init Object Counter
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=True,
                     reg_pts=region_points,
                     classes_names=model.names,
                     draw_tracks=True)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        tracks = model.track(im0, persist=True, show=False)

        im0 = counter.start_counting(im0, tracks)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    object_count_test()
