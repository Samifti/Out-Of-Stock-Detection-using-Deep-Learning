from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import os

# Define paths to your model and label files
MODEL_PATH = "/detect.tflite"
LABEL_PATH = "/labelmap.txt"

# Function to load the TFLite model and labels
def load_model():
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    with open(LABEL_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    print(f"Model loaded. Input shape: {input_details[0]['shape']}")
    print(f"Output details: {output_details}")
    return interpreter, input_details, output_details, height, width, labels

# Function to preprocess the image for the model
def preprocess_image(image, input_details, height, width):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    print(f"Image preprocessed: shape {input_data.shape}, dtype {input_data.dtype}")
    return input_data

# Function to perform object detection and draw bounding boxes
def detect_objects(image, interpreter, input_details, output_details, labels):
    input_data = preprocess_image(image, input_details, height, width)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Update the indices based on your model output details
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]

    # Print the shapes of the outputs to understand the structure
    print(f"Boxes shape: {boxes.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Classes shape: {classes.shape}")

    print(f"Boxes: {boxes}")

    # Colors for different classes
    colors = {"empty": (0, 255, 0), "disorganized": (0, 0, 255)}

    # Convert boxes to the format expected by NMSBoxes
    boxes_list = [[int(box[1] * image.shape[1]), int(box[0] * image.shape[0]), int((box[3] - box[1]) * image.shape[1]), int((box[2] - box[0]) * image.shape[0])] for box in boxes]
    scores_list = [float(score) for score in scores]

    # Filter overlapping boxes, keep the one with the highest score
    indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, score_threshold=0.1, nms_threshold=0.4)
    filtered_indices = indices.flatten() if len(indices) > 0 else []

    print(f"Detections: {len(filtered_indices)} objects detected")

    for i in filtered_indices:
        ymin, xmin, ymax, xmax = boxes[i]
        ymin = int(max(1, ymin * image.shape[0]))
        xmin = int(max(1, xmin * image.shape[1]))
        ymax = int(min(image.shape[0], ymax * image.shape[0]))
        xmax = int(min(image.shape[1], xmax * image.shape[1]))
        class_id = int(classes[i])
        class_name = labels[class_id]
        color = colors.get(class_name, (255, 255, 255))  # Default to white if class not found
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        label = f'{class_name}: {scores[i] * 100:.2f}%'
        cv2.putText(image, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        print(f"Object {i}: {label} at [{xmin}, {ymin}, {xmax}, {ymax}]")

    return image

# Function to process video frames and apply detection
def process_video(file_path, output_path, frame_skip=5):
    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0 / frame_skip, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0
    processed_frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            processed_frame = detect_objects(frame, interpreter, input_details, output_details, labels).copy()
            out.write(processed_frame)
            processed_frame_count += 1

            # Log processed frame count for debugging
            print(f"Processed frame {processed_frame_count}")

        frame_count += 1

    cap.release()
    out.release()
    print(f"Video processing completed. Total frames processed: {processed_frame_count}")

# Initialize the Flask app
app = Flask(__name__, static_folder='static')

# Load the TFLite model and labels
interpreter, input_details, output_details, height, width, labels = load_model()

@app.route('/', methods=['GET', 'POST'])
def upload_and_detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("No file part in the request")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return redirect(request.url)

        media_type = request.form['mediaType']
        if media_type == 'image':
            # Read the image file
            try:
                image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Failed to read image")

                print(f"Image uploaded: {file.filename}, shape: {image.shape}")

                # Perform object detection
                processed_image = detect_objects(image, interpreter, input_details, output_details, labels)

                # Ensure the static directory exists
                if not os.path.exists(app.static_folder):
                    os.makedirs(app.static_folder)

                # Save processed image
                save_path = os.path.join(app.static_folder, 'detected.jpg')
                cv2.imwrite(save_path, processed_image)
                print(f"Processed image saved at: {save_path}")

                # Send back the path to the processed image
                return jsonify({'media_url': url_for('static', filename='detected.jpg'), 'media_type': 'image'})
            except Exception as e:
                print(f"Error processing image: {e}")
                return redirect(request.url)

        elif media_type == 'video':
            # Save the uploaded video to a temporary location
            video_path = os.path.join(app.static_folder, 'uploaded.mp4')
            file.save(video_path)

            # Define the output path for the processed video
            output_path = os.path.join(app.static_folder, 'detected_video.mp4')

            # Process the video frames and apply object detection
            process_video(video_path, output_path)

            print(f"Processed video saved at: {output_path}")

            # Send back the path to the processed video
            return jsonify({'media_url': url_for('static', filename='detected_video.mp4'), 'media_type': 'video'})

    return render_template('upload.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)
