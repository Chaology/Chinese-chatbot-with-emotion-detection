import sys
import speech_recognition as sr
import os
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from gtts import gTTS
import time

import argparse
import cv2
import numpy as np
import webcam.face_detection_utilities as fdu
import model.myVGG as vgg
import speech_recognition as sr

import pygame

windowsName = 'Preview Screen'
FACE_SHAPE = (48, 48)
model = vgg.VGG_16('model/my_model_weights_83.h5')
emo     = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']
c = (255,255,255)


class XiaobeiBOT:


	def __init__(self):
		# self.chatbot.set_trainer(ChatterBotCorpusTrainer)
		# self.chatbot.train("chatterbot.corpus.chinese")

		self.faceDraw()

		self.chatbot = ChatBot(
	    '小贝',
	    storage_adapter = "chatterbot.storage.JsonFileStorageAdapter",
	    database = "./Xiaobei_DB.json",
	    silence_performance_warning=True
	    )
	

	def faceDraw(self):
		pygame.init()
		self.screen = pygame.display.set_mode((640,360))
		pygame.draw.rect(self.screen, c, (50,50,100,100))
		pygame.draw.rect(self.screen, c, (450,50,100,100))
		pygame.draw.rect(self.screen, c, (290,150,20,20))
		# pygame.draw.line(self.screen, c, (250,250),(350,250),10)
		pygame.draw.circle(self.screen, c, (300,250),40)
		pygame.display.update()

	def cartoonDraw(self):
		pygame.init()
		self.screen = pygame.display.set_mode((926,514))
		cartoon = pygame.image.load("face.png")
		img_rect = cartoon.get_rect()

		self.screen.fill(c)
		self.screen.blit(cartoon, img_rect)
		pygame.display.flip()


	# def mouthOpen(self):
	# 	pygame.draw.circle(self.screen, c, (300,250),40)
	# 	pygame.display.update()

	def speechRecognition(self, message = None):
		self.message = message
		r = sr.Recognizer()
		with sr.Microphone() as source:
			audio = r.listen(source)
		try:
			recContent = r.recognize_google(audio)
			print ("用户: " + recContent)
			self.message = recContent

		except sr.UnknownValueError:
			self.response = "不好意思，没有理解你在说什么"

		except sr.RequestError as e:
			self.response = "不好意思，我的大脑短路了"

	def getCameraStreaming(self):
		self.capture = cv2.VideoCapture(0)
		if not self.capture:
		    print("Failed to capture video streaming ")
		    sys.exit(1)
		else:
			print("Successed to capture video streaming")
			cv2.startWindowThread()
			cv2.namedWindow(windowsName, cv2.WND_PROP_FULLSCREEN)
			cv2.setWindowProperty(windowsName, cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)

	def greeting(self, intro):
		self.intro = intro
		self.active = True
		tts = gTTS(text = self.intro, lang = 'zh')
		tts.save('intro.mp3')
		os.system("mpg321 intro.mp3")
		print (self.intro)

	def refreshFrame(self, frame, faceCoordinates):
	    if faceCoordinates is not None:
	        fdu.drawFace(frame, faceCoordinates)
	    cv2.imshow(windowsName, frame)

	def emotionRecognition(self, message = None):
		self.message = None
		print("表情识别中...")
		cnt = 3;
		while cnt:
		    flag, frame = self.capture.read()
		    faceCoordinates = fdu.getFaceCoordinates(frame)
		    self.refreshFrame(frame, faceCoordinates)
		    
		    if faceCoordinates is not None:
		        cnt -= 1
		        face_img = fdu.preprocess(frame, faceCoordinates, face_shape=FACE_SHAPE)
		        cv2.imshow(windowsName, face_img)

		        input_img = np.expand_dims(face_img, axis=0)
		        input_img = np.expand_dims(input_img, axis=0)

		        result = model.predict(input_img)[0]
		        if cnt == 2:
		            tot_result = result
		        else:
		            tot_result += result
		        index = np.argmax(result)
		        print ('Frame',3-cnt,':', emo[index], 'prob:', max(result))
		        #index = np.argmax(result)
		        #print (emo[index], 'prob:', max(result))
		        # print(face_img.shape)
		        # emotion = class_label[result_index]
		        # print(emotion)
		index = np.argmax(tot_result)
		print ('Final decision:',emo[index], 'prob:', max(tot_result))
		self.emotion = emo[index]


	def getResponse(self):
		if self.emotion == 'Neutral':
			self.response = '开始和我聊聊天吧！'
		if self.emotion == 'Sad':
			self.response = '不要难过，有什么事情和我聊聊吧！'
		if self.emotion == 'Happy':
			self.response = '你看起来心情不错，和我聊聊天吧！'
		if self.emotion == 'Surprise':
			self.response = '你看起来很意外，和我聊聊天吧！'
		if self.emotion == 'Fear':
			self.response = '别害怕，我是小贝，和我聊聊天吧！'
		if self.emotion == 'Angry':
			self.response = '谁惹你生气了吗？和我说说'
		if self.message != None:
			self.response = str(self.chatbot.get_response(self.message))
		if len(self.response) < 1 or self.response == None:
			self.response = '我无话可说'
		if self.message == '再见':
			exit()
		tts = gTTS(text = self.response, lang = 'zh')
		tts.save('voice.mp3')
		os.system("mpg321 voice.mp3")
		print ("小贝: " + self.response)
		self.emotion = None
			

# chatbot.train("chatterbot.corpus.chinese.greetings")
# chatbot.train("chatterbot.corpus.chinese.conversations")

if __name__ == "__main__":
	bot = XiaobeiBOT()
	bot.getCameraStreaming()
	bot.greeting("你好，我叫小贝，对着镜头做个表情唤醒我吧！")
	bot.emotionRecognition()
	bot.getResponse()
	while True:
		bot.speechRecognition()
		bot.getResponse()
