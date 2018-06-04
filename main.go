package main

import (
	"fmt"
	"image/color"
	"log"
	"path/filepath"

	"github.com/mewkiz/pkg/goutil"
	"github.com/pkg/errors"
	"gocv.io/x/gocv"
)

func main() {
	if err := play(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func play() error {
	deviceID := 0

	// open webcam
	webcam, err := gocv.VideoCaptureDevice(int(deviceID))
	if err != nil {
		return errors.WithStack(err)
	}
	defer webcam.Close()

	// open display window
	win := gocv.NewWindow("Face Detect")
	defer win.Close()

	// prepare image matrix
	img := gocv.NewMat()
	defer img.Close()

	// color for the rect when faces detected
	blue := color.RGBA{0, 0, 255, 0}

	// load classifier to recognize faces
	classifier := gocv.NewCascadeClassifier()
	defer classifier.Close()

	srcDir, err := goutil.SrcDir("gocv.io/x/gocv")
	if err != nil {
		return errors.WithStack(err)
	}
	xmlFile := filepath.Join(srcDir, "data/haarcascade_frontalface_default.xml")
	if !classifier.Load(xmlFile) {
		return errors.Errorf("error reading cascade file: %q", xmlFile)
	}

	fmt.Printf("start reading camera device: %v\n", deviceID)
	for {
		if ok := webcam.Read(&img); !ok {
			return errors.Errorf("cannot read camera device %d", deviceID)
		}
		if img.Empty() {
			continue
		}

		// detect faces
		rects := classifier.DetectMultiScale(img)
		fmt.Printf("found %d faces\n", len(rects))

		// draw a rectangle around each face on the original image
		for _, r := range rects {
			gocv.Rectangle(&img, r, blue, 3)
		}

		// show the image in the window, and wait 1 millisecond
		win.IMShow(img)
		win.WaitKey(1)
	}
}
