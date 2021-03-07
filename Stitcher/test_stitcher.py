import cv2

img = cv2.imread('./000086.jpg')
while True:
    cv2.rectangle(img, (270, 125), (298, 153), (255, 0, 0), 2)
    cv2.rectangle(img, (308, 131), (356, 173), (0, 255, 0), 2)
    cv2.rectangle(img, (224, 143), (274, 182), (0, 0, 255), 2)
    cv2.imshow('image', img)

    small = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow('small', small)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
