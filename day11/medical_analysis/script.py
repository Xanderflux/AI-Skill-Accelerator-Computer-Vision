# Import Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow
import mlflow.keras
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from datetime import datetime

# Set MLFlow tracking
mlflow.set_tracking_uri("http://localhost:5001")  # Adjust if using a different MLFlow server
mlflow.set_experiment("chest_xray_pneumonia_classification")

# Set Dataset Paths, Image Dimensions, and Batch Size
train_dir = 'data/chest_xray/train'
test_dir = 'data/chest_xray/test'
val_dir = 'data/chest_xray/val'
image_size = (224, 224)
batch_size = 32

@task
def load_and_visualize_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )
    class_names = train_ds.class_names
    print("Classes:", class_names)
    
    # Visualize sample images
    for images, labels in train_ds.take(1):
        plt.figure(figsize=(8,8))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[int(labels[i])])
            plt.axis("off")
        plt.savefig("result/sample_images.png")
        mlflow.log_artifact("sample_images.png")
        plt.close()
    
    return class_names

@task
def create_data_generators():
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Preprocess and augment the training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    # Preprocess test and validation data
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    val_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    return train_generator, test_generator, val_generator

@task
def build_model():
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(base_model.input, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    
    return model

@task(cache_policy=NO_CACHE)  # Disable caching for this task
def train_model(model, train_generator, val_generator, epochs=10):
    # Ensure no active MLflow run exists
    if mlflow.active_run():
        mlflow.end_run()
    
    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("image_size", image_size)
        
        # Train model
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator
        )
        
        # Log metrics
        for epoch, (acc, val_acc) in enumerate(zip(history.history['accuracy'], 
                                                 history.history['val_accuracy'])):
            mlflow.log_metric("train_accuracy", acc, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        
        # Log model
        mlflow.keras.log_model(model, "model")
        
        return model, history

@task
def evaluate_model(model, test_generator):
    test_loss, test_accuracy = model.evaluate(test_generator)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_loss", test_loss)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')
    return test_loss, test_accuracy

@task
def save_model(model):
    model_path = 'model/cnn_model.h5'
    model.save(model_path)
    mlflow.log_artifact(model_path)
    return model_path

@task
def predict_image(model_path, image_path, class_names):
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess image
    pil_img = image.load_img(image_path, target_size=image_size)
    img_arr = image.img_to_array(pil_img)
    img_disp = img_arr.astype("uint8")
    x = np.expand_dims(img_arr, axis=0).astype("float32") / 255.0
    
    # Predict
    pred = model.predict(x, verbose=0)[0]
    prob_pneumonia = float(pred.squeeze())
    prob_normal = 1.0 - prob_pneumonia
    pred_label = class_names[int(prob_pneumonia >= 0.5)]
    true_label = os.path.basename(os.path.dirname(image_path))
    
    # Visualize prediction
    plt.figure(figsize=(5,5))
    plt.imshow(img_disp)
    plt.axis('off')
    plt.title(
        f"Predicted: {pred_label} | PNEUMONIA: {prob_pneumonia:.2%} | NORMAL: {prob_normal:.2%}\n"
        f"True label (from path): {true_label}"
    )
    plt.savefig("result/prediction.png")
    mlflow.log_artifact("result/prediction.png")
    plt.close()
    
    # Probability bar chart
    plt.figure(figsize=(4,3))
    plt.bar(class_names, [prob_normal, prob_pneumonia])
    plt.ylim(0, 1)
    plt.title("Prediction probabilities")
    plt.ylabel("Confidence")
    plt.savefig("result/probabilities.png")
    mlflow.log_artifact("result/probabilities.png")
    plt.close()

@flow(name="Pneumonia Classification Pipeline")
def pneumonia_classification_pipeline():
    class_names = load_and_visualize_data()
    train_gen, test_gen, val_gen = create_data_generators()
    model = build_model()
    model, history = train_model(model, train_gen, val_gen, epochs=1)
    evaluate_model(model, test_gen)
    model_path = save_model(model)
    predict_image(model_path, 
                 'data/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg', 
                 class_names)

if __name__ == "__main__":
    pneumonia_classification_pipeline()