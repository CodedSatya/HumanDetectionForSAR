from kivymd.uix.pickers.colorpicker.colorpicker import MDTabs
import threading
import queue
import time
import cv2
import numpy as np
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivymd.app import MDApp
from kivymd.uix.tab import MDTabsBase
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.slider import MDSlider
from kivymd.uix.textfield import MDTextField
from plyer import filechooser
from ultralytics import YOLO
import logging
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveTab(MDBoxLayout, MDTabsBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 10
        self.spacing = 10
        self.display = Image(size_hint=(1, 0.7))
        self.error_label = MDLabel(text="", halign="center", size_hint=(1, 0.1), theme_text_color="Error")
        controls = MDBoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=10)
        self.camera_btn = MDRaisedButton(text="Camera", on_press=self.start_camera)
        self.rtsp_btn = MDRaisedButton(text="RTSP", on_press=self.start_rtsp)
        self.rtsp_input = MDTextField(hint_text="Enter RTSP URL", size_hint=(0.5, None), height=40)
        self.stop_btn = MDRaisedButton(text="Stop", on_press=self.stop, disabled=True)
        controls.add_widget(self.camera_btn)
        controls.add_widget(self.rtsp_btn)
        controls.add_widget(self.rtsp_input)
        controls.add_widget(self.stop_btn)
        self.add_widget(self.display)
        self.add_widget(self.error_label)
        self.add_widget(controls)

    def start_camera(self, *args):
        self.app = MDApp.get_running_app()
        self.app.start_camera()

    def start_rtsp(self, *args):
        self.app = MDApp.get_running_app()
        self.app.start_rtsp()

    def stop(self, *args):
        self.app = MDApp.get_running_app()
        self.app.stop()

class VideoTab(MDBoxLayout, MDTabsBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 10
        self.spacing = 10
        self.display = Image(size_hint=(1, 0.7))
        self.error_label = MDLabel(text="", halign="center", size_hint=(1, 0.1), theme_text_color="Error")
        controls = MDBoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=10)
        self.load_btn = MDRaisedButton(text="Load Video", on_press=self.load_video)
        self.play_btn = MDRaisedButton(text="Play", on_press=self.play_pause)
        controls.add_widget(self.load_btn)
        controls.add_widget(self.play_btn)
        self.slider = MDSlider(min=0, max=100, value=0, size_hint=(1, 0.1))
        self.slider.bind(value=self.on_seek)
        self.time_label = MDLabel(text="0:00 / 0:00", halign="center", size_hint=(1, 0.1))
        self.add_widget(self.display)
        self.add_widget(self.error_label)
        self.add_widget(controls)
        self.add_widget(self.slider)
        self.add_widget(self.time_label)

    def load_video(self, *args):
        self.app = MDApp.get_running_app()
        self.app.load_video()

    def play_pause(self, *args):
        self.app = MDApp.get_running_app()
        self.app.play_pause()

    def on_seek(self, instance, value):
        self.app = MDApp.get_running_app()
        self.app.on_seek(instance, value)

class ImageTab(MDBoxLayout, MDTabsBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 10
        self.spacing = 10
        self.display = Image(size_hint=(1, 0.7))
        self.error_label = MDLabel(text="", halign="center", size_hint=(1, 0.1), theme_text_color="Error")
        self.load_btn = MDRaisedButton(text="Load Image", on_press=self.load_image, size_hint=(1, 0.1))
        self.add_widget(self.display)
        self.add_widget(self.error_label)
        self.add_widget(self.load_btn)

    def load_image(self, *args):
        self.app = MDApp.get_running_app()
        self.app.load_image()

class HumanDetectorApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme_cls.material_style = "M3"
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"
        model_path = 'C:\\Projects\\College Projects\\HumanDetection\\HumanDetection\\UI\\best.onnx'
        # model_path = 'C:\\Projects\\College Projects\\HumanDetection\\HumanDetection\\search_rescue_detector_100epochs_yolo11n_satellite\\human_parts_detector_100epochs_yolo11n_satellite\\weights\\best.onnx'
        self.model = YOLO(model_path, task='detect')
        self.capture = None
        self.cap = None
        self.frame = None
        self.output_frame = None
        self.selected_video = []
        self.selected_images = []
        self.total_frame = 0
        self.fps = 60
        self.is_playing = False
        self.update_event = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.last_frame_time = 0
        self.running = False
        self.last_result_time = 0
        self.frame_count = 0
        self.capture_lock = threading.Lock()
        self.cap_lock = threading.Lock()

    def build(self):
        layout = MDBoxLayout(orientation='vertical')
        self.tabs = MDTabs()
        self.live_tab = LiveTab(title="Live")
        self.video_tab = VideoTab(title="Video")
        self.image_tab = ImageTab(title="Image")
        self.tabs.add_widget(self.live_tab)
        self.tabs.add_widget(self.video_tab)
        self.tabs.add_widget(self.image_tab)
        layout.add_widget(self.tabs)
        return layout

    def on_stop(self):
        self.stop()

    def stop(self, *args):
        self.running = False
        self.is_playing = False
        if self.update_event:
            Clock.unschedule(self.update_event)
            self.update_event = None
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
        with self.capture_lock:
            if self.capture:
                self.capture.release()
                self.capture = None
        with self.cap_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        self.live_tab.camera_btn.disabled = False
        self.live_tab.rtsp_btn.disabled = False
        self.live_tab.stop_btn.disabled = True
        self.video_tab.play_btn.text = "Play"
        self.frame_queue.queue.clear()
        self.result_queue.queue.clear()
        self.live_tab.error_label.text = ""
        self.video_tab.error_label.text = ""
        self.video_tab.time_label.text = "0:00 / 0:00"

    def start_camera(self, *args):
        self.stop()
        try:
            with self.capture_lock:
                self.capture = cv2.VideoCapture(0)
                if not self.capture.isOpened():
                    self.live_tab.error_label.text = "Failed to open webcam"
                    logger.error("Failed to open webcam")
                    self.capture.release()
                    self.capture = None
                    return
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.capture.set(cv2.CAP_PROP_FPS, 60)
            time.sleep(0.1)
            self.start_processing()
        except Exception as e:
            logger.error(f"Error starting camera: {e}", exc_info=True)
            self.live_tab.error_label.text = f"Error starting camera: {str(e)}"
            with self.capture_lock:
                if self.capture:
                    self.capture.release()
                    self.capture = None

    def start_rtsp(self, *args):
        self.stop()
        rtsp_url = self.live_tab.rtsp_input.text.strip()
        if not rtsp_url.startswith('rtsp://'):
            self.live_tab.error_label.text = "Invalid RTSP URL"
            logger.error("Invalid RTSP URL")
            return
        try:
            with self.capture_lock:
                self.capture = cv2.VideoCapture(rtsp_url)
                if not self.capture.isOpened():
                    self.live_tab.error_label.text = "Failed to open RTSP stream"
                    logger.error("Failed to open RTSP stream")
                    self.capture.release()
                    self.capture = None
                    return
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.capture.set(cv2.CAP_PROP_FPS, 60)
            time.sleep(0.1)
            self.start_processing()
        except Exception as e:
            logger.error(f"Error starting RTSP: {e}", exc_info=True)
            self.live_tab.error_label.text = f"Error starting RTSP: {str(e)}"
            with self.capture_lock:
                if self.capture:
                    self.capture.release()
                    self.capture = None

    def start_processing(self):
        self.live_tab.camera_btn.disabled = True
        self.live_tab.rtsp_btn.disabled = True
        self.live_tab.stop_btn.disabled = False
        self.running = True
        self.frame_count = 0
        self.processing_thread = threading.Thread(target=self.process_live, daemon=True)
        self.processing_thread.start()
        self.update_event = Clock.schedule_interval(self.update_live, 1.0 / 30.0)
        self.live_tab.error_label.text = ""

    def update_live(self, dt):
        try:
            output_frame = self.result_queue.get_nowait()
            self.output_frame = output_frame
            self.last_result_time = time.time()
            logger.debug("Retrieved frame from result_queue for live display")
        except queue.Empty:
            if time.time() - self.last_result_time > 0.5 and self.frame is not None:
                logger.warning("Using raw frame due to empty result queue")
                self.output_frame = self.frame
            else:
                return
        if self.output_frame is not None:
            self.update_texture(self.output_frame, target='live')
            logger.debug("Updated live texture")
        
    
    def process_live(self):
        while self.running and self.capture:
            try:
                current_time = time.time()
                if current_time - self.last_frame_time < 1.0 / 60.0:
                    time.sleep(0.01)
                    continue
                self.last_frame_time = current_time
                with self.capture_lock:
                    if not self.capture or not self.capture.isOpened():
                        logger.warning("Capture closed in process_live")
                        break
                    ret, frame = self.capture.read()
                if not ret or frame is None:
                    logger.warning("Failed to read live frame")
                    continue
                if not isinstance(frame, np.ndarray) or frame.size == 0:
                    logger.warning("Invalid frame received")
                    continue
                while not self.frame_queue.empty():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
                self.frame = frame.copy()
                self.frame_count += 1
                logger.debug("Running YOLO inference on live frame")
                results = self.model(frame, conf=0.76)
                output_frame = self.draw_results(frame, results)
                logger.debug("YOLO inference completed")
                self.result_queue.put_nowait(output_frame)
                logger.debug(f"Queued frame (detected={self.frame_count % 2 == 0})")
            except queue.Full:
                logger.warning("Queue full, skipping live frame")
            except Exception as e:
                logger.error(f"Error in process_live: {e}", exc_info=True)
                self.result_queue.put_nowait(self.frame)

    def load_video(self, *args):
        filechooser.open_file(
            filters=['*.mp4', '*.avi', '*.mov'],
            on_selection=self.video_selected,
            on_exit=lambda: setattr(self.video_tab.error_label, 'text', "Video selection canceled")
        )

    def video_selected(self, selection):
        self.selected_video = selection
        if not selection:
            self.video_tab.error_label.text = "No video selected"
            logger.error("No video selected")
            return
        self.stop()
        video_path = selection[0]
        if not os.path.exists(video_path):
            self.video_tab.error_label.text = f"Video file not found: {video_path}"
            logger.error(f"Video file not found: {video_path}")
            return
        with self.cap_lock:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                self.video_tab.error_label.text = f"Failed to open video: {video_path}"
                logger.error(f"Failed to open video: {video_path}")
                self.cap.release()
                self.cap = None
                return
        self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 60
        self.video_tab.slider.max = max(self.total_frame - 1, 0)
        self.is_playing = False
        with self.cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.video_tab.error_label.text = ""
        self.video_tab.play_btn.text = "Play"
        self.update_time_label(0)

    def play_pause(self, *args):
        if not self.cap:
            self.video_tab.error_label.text = "No video loaded"
            logger.error("No video loaded")
            return
        if self.is_playing:
            self.is_playing = False
            self.video_tab.play_btn.text = "Play"
            if self.update_event:
                Clock.unschedule(self.update_event)
                self.update_event = None
            if self.processing_thread:
                self.running = False
                self.processing_thread.join(timeout=1.0)
                self.processing_thread = None
        else:
            self.is_playing = True
            self.video_tab.play_btn.text = "Pause"
            self.running = True
            self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
            self.processing_thread.start()
            self.update_event = Clock.schedule_interval(self.update_video, 1.0 / self.fps)

    def on_seek(self, instance, value):
        if self.cap:
            try:
                with self.cap_lock:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(value))
                    ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.frame = frame.copy()
                    results = self.model(frame, conf=0.65)
                    output_frame = self.draw_results(frame, results)
                    self.output_frame = output_frame
                    self.update_texture(output_frame, target='video')
                    self.update_time_label(int(value))
                else:
                    logger.warning("Failed to read frame on seek")
                    self.video_tab.error_label.text = "Failed to seek to frame"
            except Exception as e:
                logger.error(f"Error processing frame on seek: {e}", exc_info=True)
                self.video_tab.error_label.text = f"Error seeking: {str(e)}"
                self.update_texture(self.frame if self.frame is not None else np.zeros((480, 640, 3), dtype=np.uint8), target='video')

    def update_time_label(self, current_frame):
        if self.fps > 0:
            current_time = current_frame / self.fps
            total_time = self.total_frame / self.fps
            current_time_str = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
            total_time_str = f"{int(total_time // 60):02d}:{int(total_time % 60):02d}"
            self.video_tab.time_label.text = f"{current_time_str} / {total_time_str}"
        else:
            self.video_tab.time_label.text = f"Frame {current_frame} / {self.total_frame}"

    def process_video(self):
        while self.running and self.cap:
            try:
                current_time = time.time()
                if current_time - self.last_frame_time < 1.0 / self.fps:
                    time.sleep(0.01)
                    continue
                self.last_frame_time = current_time
                with self.cap_lock:
                    if not self.cap or not self.cap.isOpened():
                        logger.warning("Capture closed in process_video")
                        break
                    ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.info("Reached end of video or failed reading frame")
                    self.result_queue.put_nowait(None)
                    break
                while not self.frame_queue.empty():
                    self.frame_queue.get_nowait()
                while not self.result_queue.empty():
                    self.result_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
                self.frame = frame.copy()
                results = self.model(frame, conf=0.65)
                output_frame = self.draw_results(frame, results)
                self.result_queue.put_nowait(output_frame)
            except queue.Full:
                logger.warning("Queue full, skipping video frame")
            except Exception as e:
                logger.error(f"Error in process_video: {e}", exc_info=True)
                self.result_queue.put_nowait(self.frame if self.frame is not None else np.zeros((480, 640, 3), dtype=np.uint8))

    def update_video(self, dt):
        try:
            output_frame = self.result_queue.get_nowait()
            if output_frame is None:
                self.play_pause()
                self.video_tab.error_label.text = "Video ended"
                return
            self.output_frame = output_frame
            self.last_result_time = time.time()
        except queue.Empty:
            if time.time() - self.last_result_time > 0.5 and self.frame is not None:
                logger.warning("Using raw frame due to empty result queue")
                self.output_frame = self.frame
            else:
                return
        if self.output_frame is not None:
            self.update_texture(self.output_frame, target='video')
            with self.cap_lock:
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.cap else 0
            self.video_tab.slider.value = current_frame
            self.update_time_label(current_frame)

    def load_image(self, *args):
        filechooser.open_file(
            filters=['*.jpg', '*.jpeg', '*.png'],
            on_selection=self.image_selected,
            on_exit=lambda: setattr(self.image_tab.error_label, 'text', "Image selection canceled")
        )

    def image_selected(self, selection):
        self.selected_images = selection
        if not selection:
            self.image_tab.error_label.text = "No image selected"
            logger.error("No image selected")
            return
        image_path = selection[0]
        if not os.path.exists(image_path):
            self.image_tab.error_label.text = f"Image file not found: {image_path}"
            logger.error(f"Image file not found: {image_path}")
            return
        frame = cv2.imread(image_path)
        if frame is None:
            self.image_tab.error_label.text = f"Failed to load image: {image_path}"
            logger.error(f"Failed to load image: {image_path}")
            return
        try:
            results = self.model(frame, conf=0.29)
            output_frame = self.draw_results(frame, results)
            self.output_frame = output_frame
            self.update_texture(output_frame, target='image')
            cv2.imwrite('output.jpg', output_frame)
            # logger.info("Saved output to output.jpg")
            self.image_tab.error_label.text = ""
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            self.image_tab.error_label.text = f"Error processing image: {str(e)}"
            self.update_texture(frame, target='image')

    def draw_results(self, frame, results):
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, 'xyxy') else np.array([])
            scores = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else np.array([])
            class_ids = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, 'cls') else np.array([])
            names = result.names
            for box, score, cls_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, box[:4])
                label = f"{names[int(cls_id)]} {score:.2f}"
                if cls_id == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 3)
                elif cls_id == 1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 3)
                elif cls_id == 2:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 3)
                elif cls_id == 3:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 192, 203), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 3)
        return frame

    def update_texture(self, frame, target='live'):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.error(f"Invalid frame for {target} tab")
            return
        try:
            start_time = time.time()
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = cv2.flip(display_frame, 0)
            buf = display_frame.tobytes()
            texture = Texture.create(size=(display_frame.shape[1], display_frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            if target == 'live':
                self.live_tab.display.texture = texture
            elif target == 'video':
                self.video_tab.display.texture = texture
            elif target == 'image':
                self.image_tab.display.texture = texture
            logger.info(f"Update texture for {target} took {time.time() - start_time:.3f} seconds")
        except Exception as e:
            logger.error(f"Error updating texture for {target}: {e}", exc_info=True)

if __name__ == "__main__":
    HumanDetectorApp().run()
