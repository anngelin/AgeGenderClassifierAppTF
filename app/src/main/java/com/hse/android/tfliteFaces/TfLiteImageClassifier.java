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

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Classifies images with Tensorflow Lite.
 */
public class TfLiteImageClassifier {

  /** Tag for the {@link Log}. */
  private static final String TAG = "TfLiteCameraDemo";

    private static final int inputSize=224;

  /* Preallocated buffers for storing image data in. */
    private int[] intValues = new int[inputSize * inputSize];
    private float[][] age_outputs=new float[1][100];
    private float[][] gender_outputs=new float[1][1];
    private float[][] feature_outputs=new float[1][1024];
    private float[] prevFeatures=null;
    private Map<Integer, Object> outputMap = new HashMap<>();

    private float[][][][] img=new float[1][inputSize][inputSize][3];

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  protected Interpreter tflite;


  /** Initializes an {@code TfLiteImageClassifier}. */
  TfLiteImageClassifier(Activity activity) throws IOException {
    tflite = new Interpreter(loadModelFile(activity));
      outputMap.put(0, age_outputs);
      outputMap.put(1, gender_outputs);
      outputMap.put(2, feature_outputs);
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
      Object[] inputArray = {img};
      tflite.runForMultipleInputsOutputs(inputArray, outputMap);
    long endTime = SystemClock.uptimeMillis();
    Log.i(TAG, "tf lite timecost to run model inference: " + Long.toString(endTime - startTime));

    // Print the results.
      String textToShow = printResults();
      textToShow = Long.toString(endTime - startTime) + "ms " + textToShow;
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
      bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
      for (int i = 0; i < inputSize; ++i) {
          for (int j = 0; j < inputSize; ++j) {
              int val = intValues[j * inputSize + i];
              img[0][j][i][2] = (((val >> 16) & 0xFF) - 123.68f);
              img[0][j][i][1] = (((val >> 8) & 0xFF) - 116.779f);
              img[0][j][i][0] = ((val & 0xFF) - 103.939f);
          }
      }
  }

    /** Prints top-K labels, to be shown in UI as the results. */
    private String printResults() {
        //normalize features (first dim)
        StringBuilder str=new StringBuilder();

        float[] features=feature_outputs[0];
        float sum=0;
        for(int i=0;i<features.length;++i)
            sum+=features[i]*features[i];
        sum=(float)Math.sqrt(sum);
        for(int i=0;i<features.length;++i)
            features[i]/=sum;
        Log.i(TAG,"tf lite !!!!!!!!!!!!!!!!!!!!!!!!! end feature extraction first feat="+features[0]+" last feat="+features[features.length-1]);

        if(prevFeatures!=null){
            float dist=0;
            for(int fi=0;fi<features.length;++fi)
                dist+=(features[fi]-prevFeatures[fi])*(features[fi]-prevFeatures[fi]);
            dist/=features.length;
            str.append(String.format("dist=%.4f",dist));
        }
        else
            prevFeatures=new float[features.length];
        System.arraycopy( features, 0, prevFeatures, 0, features.length );

        //age

        //age
        final float[] age_features=age_outputs[0];
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
        str.append(" age=").append((int) Math.round(age));

        //gender
        float gender=gender_outputs[0][0];
        str.append(gender>=0.6?" male":" female").append("\n");

        Log.i(TAG,"tf lite age="+age+" gender="+gender);

        return str.toString();
    }

  /**
   * Get the name of the model file stored in Assets.
   *
   * @return
   */
  protected String getModelPath() {
    // you can download this file from
    // https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip
    return "age_gender_tf2_new-01-0.14-0.92.tflite";
    //return "mobilenet_quant_v1_224.tflite";
  }

  protected int getImageSizeX() {
        return inputSize;
    }

  protected int getImageSizeY() {
        return inputSize;
    }
}
