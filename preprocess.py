#%%
import os
import platform
import re
import shutil
system = platform.system()
print(system)
if system == "Windows":
	os.chdir("d:/School/year2/project/face_shape/")
	from config import path_cfg
else:
	os.chdir("/workspace/face_shape/")

import cv2
import pandas as pd
import matplotlib.pyplot as plt


class FACE_DATA():
	def __init__(self, cfg, del_past_img):
		self.cfg = cfg
		self.data_path = []
		self.face_path = []
		self.w_and_h = []
		self.face_cascade = cv2.CascadeClassifier(self.cfg.cv2_folder + 'haarcascade_frontalface_default.xml')
		## delete past image
		if del_past_img:
			if os.path.exists(self.cfg.save_path):
				shutil.rmtree(self.cfg.save_path)
			if not os.path.exists(self.cfg.save_path):
				os.mkdir(self.cfg.save_path)
				for label in ["round", "oval", "heart", "triangular", "square"]:
					os.mkdir(self.cfg.save_path+"/"+label)

	def get_img_path(self):
		no_support_file = ["txt", "gif", "svg"]
		for root, dirs, files in os.walk(self.cfg.data_folder):
			full_path = [
				os.path.join(root, file)
				for file in files
				if file[-3:] not in no_support_file]
			self.data_path.extend(full_path)

	def detect_face(self):
		
		system = platform.system()
		n = len(self.data_path)
		for i, path in enumerate(self.data_path):
			# print(path)		
			try:
				## read image
				img = cv2.imread(path)
				max_h = img.shape[0]
				max_w = img.shape[1]
				## to gray
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
				## detect face
				faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
				face_imgs = []
				for j, (x,y,w,h) in enumerate(faces):
					## save roi
					h = int(h*1.2)
					w1 = int(w*0.05)
					face = img[y:min(y+h, max_h), max(0, x-w1):min(x+w+w1, max_w)]
					if system == "Windows":
						ori_name = re.split(r"\\", path)[-1]
						label = re.split(r"\\", path)[-2] + "\\"
					else:
						ori_name = re.split(r"/", path)[-1]
						label = re.split(r"/", path)[-2] + "/"
					
					filename = ori_name[:-4] + "_" + str(j) + ".jpg"
					# print(self.cfg.save_path + label + filename)
					cv2.imwrite(self.cfg.save_path + label + filename, face)
					self.face_path.append(self.cfg.save_path + label + filename)
					self.w_and_h.append((w+2*w1, h))

			except Exception as err:
				print(err, path)
			if i%50==0:
				print(i, "/", n)
		## save path and w,h
		with open("face_path.txt", "w") as f:
			f.write("\n".join(self.face_path))
		pd.DataFrame(columns=["w", "h"], data=self.w_and_h).to_csv("w_h.txt", sep = " ", index = False)

	def stat(self):
		with open("face_path.txt", "r") as f:
			self.face_path = f.read().split("\n")
		print("No of data:", len(self.face_path))
		
		self.w_and_h = pd.read_table("w_h.txt", sep = " ")
		print("Mean width:", self.w_and_h["w"].mean())
		print("Mean height:", self.w_and_h["h"].mean())
		print("Mean ratio:", self.w_and_h["w"].mean()/self.w_and_h["h"].mean())
		## plot
		plt.title("Width & Height")
		plt.plot(self.w_and_h["w"], self.w_and_h["h"])
		plt.xlabel("Width")
		plt.ylabel("Height")
		# plt.plot(range(self.w_and_h.shape[0]), self.w_and_h["w"].sort_values(), label="width")
		# plt.plot(range(self.w_and_h.shape[0]), self.w_and_h["h"].sort_values(), label="height")
		# plt.legend(loc='upper left')
		plt.show()
		ratio = self.w_and_h["w"] / self.w_and_h["h"]
		plt.title("Ratio")
		plt.xlabel("Index")
		plt.ylabel("Ratio")
		plt.plot(range(self.w_and_h.shape[0]), sorted(ratio))

	def resize(self, h, w):
		pass






#%%

# folders = [item for item in os.listdir("./raw") if item[-3:]!="txt"]
cfg = path_cfg()
del_past_img = False
data = FACE_DATA(cfg, del_past_img)
if del_past_img:
	data.get_img_path()
	data.detect_face()
data.stat()

## show img
# cv2.imshow('img',img1)
# cv2.waitKey(0)a
# cv2.destroyAllWindows()
