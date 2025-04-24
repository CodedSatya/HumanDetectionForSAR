import threading

from kivy.uix.filechooser import FileChooser
from kivy.uix.accordion import Widget
from kivy.uix.image import Image
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.clock import Clock
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.textfield import MDTextField
from plyer import filechooser, orientation
from tensorflow.python.ops.signal.shape_ops import frame

from ultralytics import YOLO

from kivy.graphics.texture import Texture

import cv2

class HumanDetectorApp(MDApp):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  selected_video = []
  total_frame=0
  def build(self):
    self.frame = None
    self.output_frame = None

    self.theme_cls.material_style = "M3"
    self.theme_cls.theme_style = "Dark"
    #Importing YOLO Model
    # self.model = YOLO('C:\\Projects\\College Projects\\HumanDetection\\HumanDetection\\yolo11x.pt')
    self.model = YOLO('C:\\Projects\\College Projects\\HumanDetection\\HumanDetection\\model\\runs\detect\\train4\\weights\\best.pt')

    # self.root = Builder.load_file("HumanDetector.kv")
    # return self.root
    # self.capture = cv2.VideoCapture(0)
    # Clock.schedule_interval(self.update, 1.0/30.0)
  
  #Initializing file chooser for images
  def image_file_choser(self):
    filechooser.open_file(on_selection=self.image_selected, on_exit=self.handle_exit)

  #Definning what to do after selection of image
  def image_selected(self, selection):
    self.selected_images = selection
    if self.selected_images:
      img = cv2.imread(self.selected_images[0], cv2.IMREAD_COLOR)
      img = self.predict(img)
      buffer = cv2.flip(img, 0).tobytes()
      texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
      texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
      self.root.ids.images.texture = texture
    print(selection)

  #Handling exit without choosing
  def handle_exit(self):
    print("Choose file")

  #Handling closing opened images
  def close_images(self):
    self.selected_images = []

  #Initializing file choser for videos
  def video_file_choser(self):
    filechooser.open_file(on_selection=self.video_selected, on_exit=self.handle_exit)


  def video_selected(self, selection):
    self.selected_video = selection
    self.start_video()

  def start_video(self):
    if self.selected_video:
      self.cap = cv2.VideoCapture(self.selected_video[0])
      self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
      self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
      self.root.ids.slider.max = self.total_frame-1
      print(self.total_frame)
      self.is_playing = False

      self.update_event = None
      self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

  def vupdate(self, *args):
    ret, frame = self.cap.read()
    # print(f"Frame Read : {ret}")
    if ret:
      frame = self.predict(frame)
      frame = cv2.flip(frame, 1)

      buffer = cv2.flip(frame, 0).tobytes()
      texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
      texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')

      self.root.ids.video.texture = texture

      current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
      # print(f"Current Frame: {current_frame}")
      self.root.ids.slider.value = current_frame

      if current_frame >= self.total_frame - 1:
        self.on_play(None)
    else:
      print("End of Video or frame read failed")
      self.is_playing = False


  def video_update(self, *args):
    # ret, frame = self.cap.read()
    # if ret:
    #   frame = self.predict(frame)
    #   frame = cv2.flip(frame, 1)
    #
    #   buffer = cv2.flip(frame, 0).tobytes()
    #   texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
    #   texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
    #
    #   self.root.ids.video.texture = texture
    # else:
    #   return
    ret, frm = self.cap.read()
    if not ret:
      return
    frm = cv2.flip(frm, 0)
    frm = cv2.flip(frm, 1)
    self.frame = frm.copy()
    display_frame = self.output_frame if self.output_frame is not None else frm
    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    buff = display_frame.tobytes()
    img_txtr = Texture.create(size=(display_frame.shape[1], display_frame[0]), colorfmt='rgb')
    img_txtr.blit_buffer(buff, colorfmt='rgb', bufferfmt='ubyte')
    self.root.ids.video.texture = img_txtr


  def on_play(self, *args):
    if self.is_playing:
      self.is_playing = False
      self.root.ids.controls.text = 'Play'
      if self.update_event:
        Clock.unschedule(self.update_event)
        self.update_event = None
    else:
      self.is_playing = True
      self.root.ids.controls.text='Pause'
      # print(self.root.ids.controls.text)
      if not self.update_event:
        threading.Thread(target=self.detect, daemon=True).start()
        self.update_event = Clock.schedule_interval(self.vupdate, 1.0/self.fps)
  def on_pause(self, *args):
    pass
  def touch_down(self, *args):
    print(int(self.root.ids.slider.value))
    self.on_seek()
  def touch_move(self, *args):
    self.on_seek()

  def on_seek(self):
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.root.ids.slider.value))
    self.video_update()

  def update(self, dt):
    # ret, frame = self.capture.read()
    # if ret:
    #   self.frame = frame.copy()
    #   frame = cv2.flip(frame, 1)
    #
    #   buffer = cv2.flip(frame, 0).tobytes()
    #   texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
    #   texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
    #
    #   self.LiveCapture.texture = texture

    ret, frm = self.capture.read()
    if not ret:
      return
    frm = cv2.flip(frm, 0)
    frm = cv2.flip(frm, 1)
    self.frame = frm.copy()
    display_frame = self.output_frame if self.output_frame is not None else frm
    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    buff = display_frame.tobytes()
    img_texture = Texture.create(size=(display_frame.shape[1], display_frame.shape[0]), colorfmt='rgb')
    img_texture.blit_buffer(buff, colorfmt='rgb', bufferfmt='ubyte')
    self.LiveCapture.texture = img_texture
  def on_stop(self):
    self.capture.release()
    self.sourceSelection = MDTextField(
        hint_text="Enter Source URL",
        pos_hint={'center_y': 0.5}
      )
    self.sourceSubmit = MDRaisedButton(
      text="Go",
      pos_hint={'center_y': 0.5}
    )
    self.sourceSelectionBody = MDBoxLayout(
      orientation='horizontal',
      # pos_hint={'center_y': 0.5, 'center_x': 0.5}
      spacing="20dp"
    )

    self.sourceSelectionBody.add_widget(self.sourceSelection)
    self.sourceSelectionBody.add_widget(self.sourceSubmit)

    self.root.ids.display.clear_widgets()
    self.root.ids.display.add_widget(self.sourceSelectionBody)
  def on_start(self):
    self.LiveCapture = Image(
      size_hint= (1, 1)
    )
    self.root.ids.display.clear_widgets()
    self.root.ids.display.add_widget(self.LiveCapture)
    self.capture = cv2.VideoCapture(0)
    threading.Thread(target=self.detect, daemon=True).start()
    Clock.schedule_interval(self.update, 1.0/30.0)

  def predict(self):
    # frm = self.frame
    # results = self.model(frm)
    # for result in results:
    #   boxes = result.boxes
    #   for box in boxes:
    #     x1, y1, x2, y2 = map(int, box.xyxy[0])
    #     confidence = box.conf[0]
    #     class_id = int(box.cls[0])
    #     label = f"{self.model.names[class_id]} {confidence:.2f}"
    #
    #     cv2.rectangle(frm, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(frm, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 2)
    # return frm
    if self.frame is None:
      return
    frm = self.frame.copy()
    # frm = cv2.flip(frm, 0)
    results = self.model(source=frm, conf=0.65)
    for result in results:
      boxes = result.boxes
      for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        label = f'{self.model.names[class_id]} {confidence:.2f}'
        cv2.rectangle(frm, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frm, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    self.output_frame = frm
  def detect(self):
    while True:
      if self.frame is not None:
        self.predict()
if __name__ == "__main__":
  HumanDetectorApp().run()