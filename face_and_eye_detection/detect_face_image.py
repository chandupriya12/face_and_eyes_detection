import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('test.jpg')

# Resize the image while maintaining the aspect ratio
scale_factor = 0.5  # Adjust this value to control the size of the displayed image
dim = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# Convert to grayscale
gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(resized_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output
cv2.imshow('img', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


