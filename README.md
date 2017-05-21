# Simple script to detect and recognize plate number (Automatic Plate Number Recognition - APNR)

**[Link to the .pdf report](https://raw.githubusercontent.com/kagan94/Automatic-Plate-Number-Recognition-APNR/master/Report.pdf)**

## Steps:
1) Load image
2) Apply **_blur filter_** (to remove noise)
3) Convert blurred image to grayscale
4) Apply **_Sobel filter_** to find vertical edges (car plates have a high density of vertical lines)
5) Apply threshold with Ostu’s Binarization
	(Ostu’s binarization will automatically calculate optimal threshold from image histogram)

	![steps 1-5](https://raw.githubusercontent.com/kagan94/Automatic-Plate-Number-Recognition-APNR/master/report_imgs/steps_1-5.png)
 
6) Create a rectangular mask of size of 17x3 and apply “closing” filter to detect plate number more clearly

![steps 1-5](https://raw.githubusercontent.com/kagan94/Automatic-Plate-Number-Recognition-APNR/master/report_imgs/step6.png)

7) Find and fetch contours of possible plates
8) Validate contours and clear out those, that can't be potential plate numbers
	- Is white color dominant?
	- Rotated not more than 15 degrees
	- In Europe, car plate size: 52x11, aspect 4,7272
	- Define min && max area of plate number
9) After (8), apply **_dilate filter_** and threshold to validated contours to get numbers and characters

![steps 7-9](https://raw.githubusercontent.com/kagan94/Automatic-Plate-Number-Recognition-APNR/master/report_imgs/steps_7-9.png)

![cleaning plate number](https://raw.githubusercontent.com/kagan94/Automatic-Plate-Number-Recognition-APNR/master/report_imgs/cleaning_plate_number.png)

10)Apply **Tesseract** to extract plate number as a text. Tesseract is an optical character recognition (OCR) engine sponsored by Google.

![steps 10](https://raw.githubusercontent.com/kagan94/Automatic-Plate-Number-Recognition-APNR/master/report_imgs/resseract_result.png)

### Final Results
![Final Results](https://raw.githubusercontent.com/kagan94/Automatic-Plate-Number-Recognition-APNR/master/report_imgs/final_result.png)
