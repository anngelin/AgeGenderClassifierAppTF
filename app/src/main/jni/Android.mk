LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#opencv
#OPENCVROOT:= /Users/avsavchenko/Documents/my_soft/github/opencv/opencv/platforms/build_android_arm/install/
OPENCVROOT:= /Users/avsavchenko/Documents/my_soft/github/opencv/OpenCV-android-sdk
OPENCV_CAMERA_MODULES:=off
OPENCV_INSTALL_MODULES:=on
OPENCV_LIB_TYPE:=SHARED
include ${OPENCVROOT}/sdk/native/jni/OpenCV.mk

LOCAL_SRC_FILES := com_hse_android_tfliteFaces_DetectionBasedTracker.cpp
LOCAL_CFLAGS += -mfloat-abi=softfp -mfpu=neon -std=c++11 # -march=armv7
LOCAL_ARM_NEON  := true
LOCAL_LDLIBS += -llog
LOCAL_MODULE := OpenCvDetectionLib

include $(BUILD_SHARED_LIBRARY)