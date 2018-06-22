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
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

/**
 * Classifies images with Tensorflow Lite.
 */
public class AgeGenderTfMobileClassifier {

  /** Tag for the {@link Log}. */
  private static final String TAG = "TfMobileClassifier";

  private static final int DIM_PIXEL_SIZE = 3;

    private TensorFlowInferenceInterface inferenceInterface;

    /* Preallocated buffers for storing image data in. */
    private int[] intValues = new int[getImageSizeX() * getImageSizeY()];
    private float[] floatValues=new float[getImageSizeX() * getImageSizeY()*DIM_PIXEL_SIZE];
    private float[][] outputs;
    private float[] prevFeatures=null;

    private static final String INPUT_NAME = "input_1";
    private static final String[] OUTPUT_NAMES = {"global_pooling/Mean","age_pred/Softmax","gender_pred/Sigmoid"};
    private static final String MODEL_FILE =
            "file:///android_asset/age_gender_tf2_new-01-0.14-0.92.pb";
            //"file:///android_asset/optimized_quantized_graph.pb";

  /** Initializes an {@code TfLiteImageClassifier}. */
  AgeGenderTfMobileClassifier(Activity activity) throws IOException {
      inferenceInterface = new TensorFlowInferenceInterface(activity.getAssets(),MODEL_FILE);
      outputs = new float[OUTPUT_NAMES.length][];
      for(int i=0;i<OUTPUT_NAMES.length;++i) {
          String featureOutputName = OUTPUT_NAMES[i];
          // The shape of the output is [N, NUM_OF_FEATURES], where N is the batch size.
          int numOFFeatures = (int) inferenceInterface.graph().operation(featureOutputName).output(0).shape().size(1);
          Log.i(TAG, "Read output layer size is " + numOFFeatures);
          outputs[i] = new float[numOFFeatures];
      }
    Log.d(TAG, "Created a Tensorflow Mobile Image Classifier.");
  }

  /** Classifies a frame from the preview stream. */
  String classifyFrame(Bitmap bitmap) {
      long startTime = SystemClock.elapsedRealtime();
      convertBitmapToByteBuffer(bitmap);
    runInference();
    long endTime = SystemClock.elapsedRealtime();
    Log.i(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

    // Print the results.
    String textToShow = printResults();
    textToShow = Long.toString(endTime - startTime) + "ms " + textToShow;
    return textToShow;
  }

  /** Closes tflite to release resources. */
  public void close() {
    inferenceInterface.close();
  }


  /** Writes Image data into a {@code ByteBuffer}. */
  private void convertBitmapToByteBuffer(Bitmap bitmap) {
      bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
      for (int i = 0; i < intValues.length; ++i) {
          final int val = intValues[i];
          //'RGB'->'BGR'
          floatValues[i * 3 + 0] = ((val & 0xFF) - 103.939f);
          floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - 116.779f);
          floatValues[i * 3 + 2] = (((val >> 16) & 0xFF) - 123.68f);
      }
  }

  /** Prints top-K labels, to be shown in UI as the results. */
  private String printResults() {
      //normalize features (first dim)
      StringBuilder str=new StringBuilder();

      float[] features=outputs[0];
      float sum=0;
      for(int i=0;i<features.length;++i)
          sum+=features[i]*features[i];
      sum=(float)Math.sqrt(sum);
      for(int i=0;i<features.length;++i)
          features[i]/=sum;
      Log.i(TAG,"!!!!!!!!!!!!!!!!!!!!!!!!! end feature extraction first feat="+features[0]+" last feat="+features[features.length-1]);

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
      final float[] age_features=outputs[1];
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
      float gender=outputs[2][0];
      str.append(gender>=0.6?" male":" female").append("\n");

      Log.i(TAG,"age="+age+" gender="+gender);

    return str.toString();
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



  protected void runInference() {
      inferenceInterface.feed(INPUT_NAME, floatValues, 1, getImageSizeX(), getImageSizeY(), DIM_PIXEL_SIZE);
      inferenceInterface.run(OUTPUT_NAMES);

      // Copy the output Tensor back into the output array.
      for(int i=0;i<OUTPUT_NAMES.length;++i) {
          inferenceInterface.fetch(OUTPUT_NAMES[i], outputs[i]);
      }

  }

}
