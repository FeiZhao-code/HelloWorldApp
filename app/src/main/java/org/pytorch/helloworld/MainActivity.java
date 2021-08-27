package org.pytorch.helloworld;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.StrictMode;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.MemoryFormat;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

  Module module = null;
  Bitmap bitmap = null;
  int bitmap_index = 1;
  private ImageView imageView;
  private TextView textView;
  private ArrayList<String> classNames;



  public static void closeStrictMode() {
    StrictMode.setThreadPolicy(new StrictMode.ThreadPolicy.Builder()
            .detectAll().penaltyLog().build());
  }
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    try {

      module = LiteModuleLoader.load(assetFilePath(this, "m24.pt"));
    } catch (IOException e) {
      e.printStackTrace();
    }
    // 获取控件
    imageView = findViewById(R.id.image);
    textView = findViewById(R.id.text);
    Button selectImgBtn = findViewById(R.id.select_img_btn);
    Button openCamera = findViewById(R.id.open_camera);

    selectImgBtn.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        //
        try {
          bitmap = BitmapFactory.decodeStream(getAssets().open(bitmap_index+".jpg"));
        } catch (IOException e) {
          e.printStackTrace();
        }
        if(bitmap_index==11)
          bitmap_index=0;
        bitmap_index++;
        imageView.setImageBitmap(bitmap);

        // preparing input tensor
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);

        // running the model
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        // getting tensor content as java array of floats
        final float[] scores = outputTensor.getDataAsFloatArray();

        // searching for the index with maximum score
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
          if (scores[i] > maxScore) {
            maxScore = scores[i];
            maxScoreIdx = i;
          }
        }

        String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];

        // showing className on UI
        TextView textView = findViewById(R.id.text);
        textView.setText(className);
      }
    });
    openCamera.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        // 打开实时拍摄识别页面
        Intent intent = new Intent(MainActivity.this, CameraActivity.class);
        startActivity(intent);
      }
    });

  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */


  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }
}
