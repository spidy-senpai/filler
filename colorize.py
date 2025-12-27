def fillcolour_model(image): 

	"""
		Credits: 
			1. https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py
			2. http://richzhang.github.io/colorization/
			3. https://github.com/richzhang/colorization/
		"""

	# Import statements
	import numpy as np
	import argparse
	import cv2
	import os
	from PIL  import Image 

	"""
	Download the model files: 
		1. colorization_deploy_v2.prototxt:    https://github.com/richzhang/colorization/tree/caffe/colorization/models
		2. pts_in_hull.npy:					   https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
		3. colorization_release_v2.caffemodel: https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1

	"""

	# Paths to load the model
	DIR = os.path.dirname(os.path.abspath(__file__))
	PROTOTXT = os.path.join(DIR, "model", "colorization_deploy_v2.prototxt")
	POINTS = os.path.join(DIR, "model", "pts_in_hull.npy")
	
	# Use /tmp for Vercel (writable), fall back to local model dir for development
	import tempfile
	TEMP_MODEL_DIR = os.path.join(tempfile.gettempdir(), "colorize_model")
	MODEL = os.path.join(TEMP_MODEL_DIR, "colorization_release_v2.caffemodel")

	# Download model from Google Drive if it doesn't exist
	if not os.path.exists(MODEL):
		print("Model not found locally. Downloading from Google Drive...")
		import requests
		
		# Google Drive direct download URL
		GOOGLE_DRIVE_URL = "https://drive.google.com/uc?export=download&id=1wk_gADxhzMfvW_tnb1voCSEB2oJ3ARP_"
		
		try:
			# Create model directory if it doesn't exist (in /tmp which is writable)
			os.makedirs(TEMP_MODEL_DIR, exist_ok=True)
			
			print(f"Starting download from Google Drive to {TEMP_MODEL_DIR}...")
			session = requests.Session()
			
			# Download with a longer timeout for large files
			response = session.get(GOOGLE_DRIVE_URL, stream=True, timeout=600, verify=False)
			response.raise_for_status()
			
			# Get total file size
			total_size = int(response.headers.get('content-length', 0))
			print(f"Downloading {total_size / (1024*1024):.1f} MB...")
			
			# Write the file in chunks
			downloaded = 0
			with open(MODEL, 'wb') as f:
				for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
					if chunk:
						f.write(chunk)
						downloaded += len(chunk)
						if downloaded % (10*1024*1024) == 0:  # Log every 10MB
							print(f"Downloaded {downloaded / (1024*1024):.1f} MB...")
			
			print(f"Model downloaded successfully! File size: {os.path.getsize(MODEL) / (1024*1024):.1f} MB")
		except Exception as e:
			error_msg = f"Failed to download model: {str(e)}"
			print(f"ERROR: {error_msg}")
			raise FileNotFoundError(error_msg)

	# Argparser
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--image", type=str, required=True,
	# 	help="path to input black and white image")
	# args = vars(ap.parse_args())

	# Load the Model
	print("Load model")
	net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
	pts = np.load(POINTS)

	# Load centers for ab channel quantization used for rebalancing.
	class8 = net.getLayerId("class8_ab")
	conv8 = net.getLayerId("conv8_313_rh")
	pts = pts.transpose().reshape(2, 313, 1, 1)
	net.getLayer(class8).blobs = [pts.astype("float32")]
	net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

	# Load the input image
	# image = cv2.imread(image)
	pil_image = Image.open(image)
	image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
	scaled = image.astype("float32") / 255.0
	lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)


	resized = cv2.resize(lab, (224, 224))
	L = cv2.split(resized)[0]
	L -= 50

	print("Colorizing the image")
	net.setInput(cv2.dnn.blobFromImage(L))
	ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

	ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

	L = cv2.split(lab)[0]
	colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

	colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
	colorized = np.clip(colorized, 0, 1)

	colorized = (255 * colorized).astype("uint8")
	colorized_pil = Image.fromarray(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB))
	print("succesful")
	return colorized_pil


