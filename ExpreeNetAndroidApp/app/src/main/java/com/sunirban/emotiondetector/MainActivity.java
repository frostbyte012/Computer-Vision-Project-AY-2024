package com.sunirban.emotiondetector;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.gms.tasks.Task;
import com.google.firebase.FirebaseApp;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;


public class MainActivity extends AppCompatActivity {

    Button button;
    TextView textView;
    ImageView imageView;

    String[] emotion_labels = {"Angry", "Disgust","Fear", "Happy", "Neutral", "Sad", "Surprise"};

    Interpreter tflite;
    FaceDetector detector;
    Paint paint;

    private final ActivityResultLauncher<Intent> captureImageResultLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Bundle extras = result.getData().getExtras();
                    assert extras != null;
                    Bitmap bitmap = (Bitmap) extras.get("data");
                    imageView.setImageBitmap(bitmap);
                    FaceDetectionProcess(bitmap);
                    Toast.makeText(this, "Success!", Toast.LENGTH_SHORT).show();
                }
            }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        button = findViewById(R.id.camera_btn);
        textView = findViewById(R.id.text1);
        imageView = findViewById(R.id.imageView);

        Animation fadeIn = AnimationUtils.loadAnimation(this, R.anim.fade_in);
        textView.startAnimation(fadeIn);

        Animation slideUp = AnimationUtils.loadAnimation(this, R.anim.slides_up);
        button.startAnimation(slideUp);

        FirebaseApp.initializeApp(this);

        button.setOnClickListener(view -> OpenFile());
        Toast.makeText(this, "App is Started!", Toast.LENGTH_SHORT).show();

        try {
            tflite = new Interpreter(loadModelFile());
        } catch (Exception e) {
            e.printStackTrace();
        }

        FaceDetectorOptions highAccuracyOption = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                .enableTracking().build();

        detector = FaceDetection.getClient(highAccuracyOption);

        paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(1);
        paint.setTextSize(7);
    }

    private void OpenFile() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (intent.resolveActivity(getPackageManager()) != null){
            captureImageResultLauncher.launch(intent);
        } else {
            Toast.makeText(this, "Failed!", Toast.LENGTH_SHORT).show();
        }
    }

    private void FaceDetectionProcess(Bitmap bitmap) {
        textView.setText(R.string.processing_image);
        final StringBuilder builder = new StringBuilder();

        InputImage image = InputImage.fromBitmap(bitmap, 0);

        Task<List<Face>> result = detector.process(image);
        result.addOnSuccessListener(faces -> {
            if(!faces.isEmpty()){
                if(faces.size() == 1){
                    builder.append(faces.size()).append(" Face Detected\n\n");
                } else {
                    builder.append(faces.size()).append(" Faces Detected\n\n");
                }
            }
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            Canvas canvas = new Canvas(mutableBitmap);

            for (Face face: faces){
                canvas.drawRect(face.getBoundingBox(), paint);
                int id = face.getTrackingId();
                canvas.drawText("ID: " + id, face.getBoundingBox().left, face.getBoundingBox().top, paint);
                float rotY = face.getHeadEulerAngleY();
                float rotZ = face.getHeadEulerAngleZ();

                builder.append("1. Face Tracking ID [").append(id).append("]\n");
                builder.append("2. Head Rotation to right [").append(String.format("%.2f", rotY)).append(" deg. ]\n");
                builder.append("3. Head Tilted Sideways [").append(String.format("%.2f", rotZ)).append(" deg. ]\n");

                // Smiling
                if(face.getSmilingProbability()>0){
                    float smilingProbability = face.getSmilingProbability();
                    builder.append("4. Smiling Probability [").append(String.format("%.2f", smilingProbability)).append("]\n");
                }


                // Preprocess the face image
                Bitmap faceBitmap = Bitmap.createBitmap(bitmap, face.getBoundingBox().left, face.getBoundingBox().top, face.getBoundingBox().width(), face.getBoundingBox().height());
                Bitmap grayscaleBitmap = toGrayscale(faceBitmap);
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(grayscaleBitmap, 48, 48, true);
                float[][][][] input = new float[1][48][48][1];
                for (int x = 0; x < 48; x++) {
                    for (int y = 0; y < 48; y++) {
                        input[0][x][y][0] = (resizedBitmap.getPixel(x, y) & 0xFF) / 255.0f;
                    }
                }

                // Load the model
                try {
                    float[][] output = new float[1][7];

                    // Run inference
                    tflite.run(input, output);

                    for (int i = 0; i < output[0].length; i++) {
                        Log.d("Model Output", "Probability of " + emotion_labels[i] + ": " + output[0][i]);
                    }

                    // Postprocess the results
                    int predictedEmotion = argmax(output[0]);
                    builder.append("5. Emotion: ").append(emotion_labels[predictedEmotion]).append("\n");
                } catch (Exception e) {
                    e.printStackTrace();
                }

                builder.append("\n");

            }
            imageView.setImageBitmap(mutableBitmap);
            ShowDetection(builder, true);
        }).addOnFailureListener(e -> {
            StringBuilder builder1 = new StringBuilder();
            builder1.append("Sorry! There is an error!");
            ShowDetection(builder1, false);
        });

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        detector.close();
        tflite.close();
    }

    private void ShowDetection(final StringBuilder builder, boolean success) {
        if (success){
            textView.setText(null);
            textView.setMovementMethod(new ScrollingMovementMethod());

            if(builder.length() !=0){
                textView.append(builder);
                textView.append("(Hold the text to copy it!)");

                textView.setOnLongClickListener(view -> {
                    ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
                    ClipData clip = ClipData.newPlainText("Face Detection", builder);
                    clipboard.setPrimaryClip(clip);
                    return true;
                });
            } else {
                textView.append("Failed to detect Anything!");
            }
        } else {
            textView.setText(null);
            textView.setMovementMethod(new ScrollingMovementMethod());
            textView.append(builder);
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        MappedByteBuffer mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        fileChannel.close();
        inputStream.close();

        return mappedByteBuffer;
    }

    private Bitmap toGrayscale(Bitmap bmpOriginal) {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        return bmpGrayscale;
    }

    private int argmax(float[] probabilities) {
        float total = 0.0f;
        for (float prob : probabilities) {
            total += (float) Math.exp(prob);
        }
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = (float) Math.exp(probabilities[i]) / total;
        }

        int best = 0;
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }
}