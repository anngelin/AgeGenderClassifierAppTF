/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.hse.android.tfliteFaces;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;

/**
 * Classifies images with Tensorflow Lite.
 */
public class AgeGenderClassifier {

  /** Tag for the {@link Log}. */
  private static final String TAG = "AgeGenderClassifier";

  private static final int DIM_PIXEL_SIZE = 3;

  /* Preallocated buffers for storing image data in. */
  private int[] intValues = new int[getImageSizeX() * getImageSizeY()];

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  protected Interpreter tflite;

  /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
  protected ByteBuffer imgData = null;

    private float[][] ageProbArray = null,genderSigmoidArray=null,featuresArray=null;
    private Map<Integer, Object> cnnOutputs = new HashMap<>();

  /** Initializes an {@code TfLiteImageClassifier}. */
  AgeGenderClassifier(Activity activity) throws IOException {
    tflite = new Interpreter(loadModelFile(activity));
    imgData =
        ByteBuffer.allocateDirect(
            getImageSizeX()
                * getImageSizeY()
                * DIM_PIXEL_SIZE
                * getNumBytesPerChannel());
    imgData.order(ByteOrder.nativeOrder());

      //age
      ageProbArray=new float[1][100];
      cnnOutputs.put(0,ageProbArray);
      //gender
      genderSigmoidArray=new float[1][1];
      cnnOutputs.put(1,genderSigmoidArray);
      //features
      featuresArray=new float[1][1024];
      cnnOutputs.put(2,featuresArray);
    Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
  }

  /** Classifies a frame from the preview stream. */
  String classifyFrame(Bitmap bitmap) {
    if (tflite == null) {
      Log.e(TAG, "Image classifier has not been initialized; Skipped.");
      return "Uninitialized Classifier.";
    }
    convertBitmapToByteBuffer(bitmap);
    // Here's where the magic happens!!!
    long startTime = SystemClock.uptimeMillis();
    runInference();
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

    // Print the results.
    String textToShow = printResults();
    textToShow = Long.toString(endTime - startTime) + "ms" + textToShow;
    return textToShow;
  }

  /** Closes tflite to release resources. */
  public void close() {
    tflite.close();
    tflite = null;
  }

  /** Memory-map the model file in Assets. */
  private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /** Writes Image data into a {@code ByteBuffer}. */
  private void convertBitmapToByteBuffer(Bitmap bitmap) {
    if (imgData == null) {
      return;
    }
    imgData.rewind();
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    // Convert the image to floating point.
    int pixel = 0;
    long startTime = SystemClock.uptimeMillis();
    for (int i = 0; i < getImageSizeX(); ++i) {
      for (int j = 0; j < getImageSizeY(); ++j) {
        final int val = intValues[pixel++];
        addPixelValue(val);
      }
    }
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
  }

  /** Prints top-K labels, to be shown in UI as the results. */
  private String printResults() {
      //normalize features (first dim)
      StringBuilder str=new StringBuilder();
      float sum=0;
      for(int i=0;i<featuresArray[0].length;++i)
          sum+=featuresArray[0][i]*featuresArray[0][i];
      sum=(float)Math.sqrt(sum);
      for(int i=0;i<featuresArray[0].length;++i)
          featuresArray[0][i]/=sum;
      Log.i(TAG,"!!!!!!!!!!!!!!!!!!!!!!!!! end feature extraction first feat="+featuresArray[0][0]+" last feat="+featuresArray[0][featuresArray[0].length-1]);

      //age
      final float[] age_features=ageProbArray[0];
      ArrayList<Integer> indices = new ArrayList<>();
      for (int j=0;j<age_features.length;++j){
          indices.add(j);
      }
      Collections.sort(indices, new Comparator<Integer>() {
          @Override
          public int compare(Integer idx1, Integer idx2) {
              if (age_features[idx1]==age_features[idx2])
                  return 0;
              else if (age_features[idx1]>age_features[idx2])
                  return -1;
              else
                  return 1;
          }
      });
      int max_index=2;
      float[] probabs=new float[max_index];
      sum=0;
      for(int j=0;j<max_index;++j){
          probabs[j]=age_features[indices.get(j)];
          sum+=probabs[j];
      }
      double age=0;
      for(int j=0;j<max_index;++j) {
          age+=(indices.get(j)+0.5)*probabs[j]/sum;
      }
      str.append("age=").append((int) Math.round(age));

      //gender
      float gender=genderSigmoidArray[0][0];
      str.append(gender>=0.6?" male":" female");

    return str.toString();
  }


  protected String getModelPath() {
    //return "mobilenet_quant_v1_224.tflite";
      return "age_gender_tf2_new-01-0.14-0.92.tflite";
  }

  /**
   * Get the image size along the x axis.
   *
   * @return
   */
  protected int getImageSizeX() {
    return 224;
  }

  /**
   * Get the image size along the y axis.
   *
   * @return
   */
  protected int getImageSizeY() {
    return 224;
  }

  /**
   * Get the number of bytes that is used to store a single color channel value.
   *
   * @return
   */
  protected int getNumBytesPerChannel() {
    // the quantized model uses a single byte only
    //return 1;
    // a 32bit float value requires 4 bytes
    return 4;
  }

  /**
   * Add pixelValue to byteBuffer.
   *
   * @param val
   */
  protected void addPixelValue(int val) {
    /*imgData.put((byte) ((val >> 16) & 0xFF));
    imgData.put((byte) ((val >> 8) & 0xFF));
    imgData.put((byte) (val & 0xFF));
    */

      imgData.putFloat(((val >> 16) & 0xFF) - 123.68f);
      imgData.putFloat(((val >> 8) & 0xFF) - 116.779f);
      imgData.putFloat((val & 0xFF) - 103.939f);

  }

  /**
   * Run inference using the prepared input in {@link #imgData}. Afterwards, the result will be
   * provided by getProbability().
   *
   * <p>This additional method is necessary, because we don't have a common base for different
   * primitive data types.
   */
  protected void runInference() {

      Object[] inputs = {imgData};
      tflite.runForMultipleInputsOutputs(inputs, cnnOutputs);
  }

}
