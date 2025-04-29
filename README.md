# 📄 Project Documentation: End-to-End Self-Driving Car Simulation

## 1. Project Title
DeepDrive: End-to-End Autonomous Driving in a Simulated Environment

## 2. Project Overview
This project demonstrates an end-to-end deep learning approach to autonomous driving by training a Convolutional Neural Network (CNN) that predicts steering angles directly from camera images.

* The system learns driving behavior without manual feature engineering and operates within a simulation environment (e.g., Udacity Self-Driving Car Simulator).
* The goal is to show how modern self-driving car systems can use computer vision and deep learning to make driving decisions.

## Implementation : 

<div style="display: flex; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/83ea9528-1b5a-4995-8edd-a1f56bfb9296" width="45%">
  <img src="https://github.com/user-attachments/assets/2e648c68-d3f6-4821-bef6-654ee89731ce" width="45%">
</div>


## 3. What's New / Highlights
* **End-to-End Pipeline:** Raw images to steering commands — no manual lane detection or traditional CV techniques.
* ~**Data Augmentation:** Improves model robustness (e.g., brightness shifts, flips, camera angle offsets).~
* **Behavioral Cloning:** Mimics human driving behaviors by learning from recorded expert demonstrations.
* **Model Interpretability:** Saliency maps and activation visualizations show what the model "sees."
* **Optional Extension:** Reinforcement Learning (DQN/Policy Gradients) for driving without imitation.

## 4. Technologies and Tools Used
| Category          | Tools/Frameworks                                       |
| :---------------- | :----------------------------------------------------- |
| Deep Learning     | TensorFlow & Keras                                     |
| Computer Vision   | OpenCV                                                 |
| Simulator         | Udacity Self-Driving Car Simulator                     |
| Data Handling     | NumPy, Pandas                                          |
| Visualization     | Matplotlib, Seaborn                                    |
| Working           | mss (capture screenshots), pyautogui (keyboard inputs) |

## 5. Architecture
Here’s the end-to-end flow:

````
flowchart LR
    A[Simulator (Camera Images + Steering Data)] --> B[Data Preprocessing]
    B --> C[Data Augmentation]
    C --> D[CNN Model Training]
    D --> E[Model Evaluation (Validation set)]
    E --> F[Real-time Prediction in Simulator]
````

CNN Model Architecture:
(Adapted from Nvidia’s End-to-End Self-Driving Paper)

* **Input:** 3-channel RGB images (cropped and resized)
* **5 Convolutional Layers:**
    * (24 filters, 5x5 kernel, stride 2)
    * (36 filters, 5x5 kernel, stride 2)
    * (48 filters, 5x5 kernel, stride 2)
    * (64 filters, 3x3 kernel)
    * (64 filters, 3x3 kernel)
* Flatten
* **3 Fully Connected Layers:**
    * (100 → 50 → 10 → 1 output: Steering Angle)
 
  <p align="center">
  <img src="https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png" alt="Nvidia Paper Fig 5">
</p>

## 6. Data Collection and Preprocessing
* **Data Source:**
    * Record your own driving inside the simulator (center, left, right camera views + steering angle).
* **Preprocessing Steps:**
    * Crop irrelevant parts (sky, dashboard)
    * Normalize pixel values (between -1 and 1)
    * Resize to a smaller input shape (e.g., 66x200)
* **Data Augmentation:**
    * Random brightness change
    * Random shadow addition
    * Horizontal flipping (invert steering)
    * Slight shifts in width and height
    * Adjust steering for side cameras

## 7. Training Strategy
* **Loss function:** Mean Squared Error (MSE) (between predicted and actual steering angle)
* **Optimizer:** Adam (with tuned learning rate, e.g., 1e-4)
* **Batch Size:** 32–64
* Early stopping and checkpointing to avoid overfitting
* Model trained for behavior cloning (learning from a human driver)

## 8. Evaluation Metrics
* Training and Validation Loss Curves
* **On-road performance in the simulator:**
    * No lane departures
    * Smooth steering without jerky motions
* **Optional Metrics:**
    * Time without human intervention
    * Average deviation from the center of the lane

## 9. Challenges and Solutions
| Challenge                         | Solution                                                       |
| :-------------------------------- | :------------------------------------------------------------- |
| Overfitting on Training Data      | Data augmentation, dropout layers                              |
| Poor Generalization on Curves     | Include more sharp turns and recovery data                     |
| Model drifting over time          | Train on side camera images with corrected steering angles     |


### 9.1 Major challenges **Still Facing**
1. The simulators webscket not woking hence the screenshot approach used
2. Due to point 1 training screenshots and testing screenshots differ and hence the mode steering suffers

**Solutions**
## 🛣 Idea 1: Changing the Screenshot Area
**Summary:**

Adjust the screen capture (`monitor = {...}`) to mimic the dashcam view (lower FOV, lower angle).
**Feasibility:** ✅✅✅

Highly feasible, super easy to do.
Capturing a smaller, properly placed part of the screen will reduce domain gap between training and testing.
Dashcams usually see only front, not the sky or too much side view — so narrowing capture area makes your model "feel at home."
You can keep tuning `top`, `left`, `width`, `height` until it looks similar to the original training images.
**Bonus Tip:**

Also try applying a slight Gaussian blur or color adjustment to the captured image to mimic the camera quality (because screens are sharper than dashcams).

## 🚗 Idea 2: Create New Dataset by Driving Manually
**Summary:**

While you drive, capture:
* Screenshot images
* Associated steering labels (`a` = left, `d` = right, `w` = straight, `s` = backward)

and save them as CSV.
**Feasibility:** ✅✅✅✅✅ (and smart)

This is the **BEST** way to create your own domain-specific training data.
You’ll eliminate domain mismatch 100%.
Capturing your own third-person view can even allow you to train for better planning (not just steering).
CSV with `image_name`, `steering_angle` is industry standard format.
**You will need:**

* `keyboard` or `pynput` library to read key presses.
* `PIL` or `cv2.imwrite()` to save screenshots.
* Small function to map keys:
    ```
    w = 0.0
    a = -1.0
    d = +1.0
    s = reverse? maybe a special tag or very negative value
    ```
**Example CSV:**
```csv
image_name,steering_angle
00001.jpg,0.0
00002.jpg,-0.5
00003.jpg,1.0
```

## ⚡ Final verdict:

| Idea                     | Feasibility | My Comment                                                      |
| ------------------------ | ----------- | --------------------------------------------------------------- |
| Change Screenshot Area   | ✅          | Easy and quick fix. **Do this immediately and tune.** |
| Collect Own Dataset      | ✅✅        | Best but needs small project. **Totally worth the time if you want proper retraining.** |

***Will Try these solutions in the next few iterations***

## 10. Possible Extensions
* Reinforcement Learning agent to learn driving policy without supervision
* Obstacle Detection using Object Detection Models (YOLOv8)
* Multi-task Learning (predict steering + throttle + brake)
* Deploy on a real RC car (NVIDIA Jetson Nano + Raspberry Pi)

## 11. Key Takeaways for Resume
* Built an end-to-end autonomous driving model from scratch
* Demonstrated skills in CNNs, Computer Vision, Simulation Control, and Model Interpretability
* Showed the ability to handle real-world-like imperfections using data augmentation
* Delivered a visually impressive, real-time deep learning system

✅Pesonal tips:
* Create a short demo video (30 seconds to 2 minutes) showing:
* The simulator recording phase
* The model training phase (short glimpse)
* The car driving autonomously in the simulator

## Reference Paper
[1]  Mariusz Bojarski, Ben Firner, Beat Flepp, Larry Jackel, Urs Muller, Karol Zieba and Davide Del Testa. *End-to-End Deep Learning for Self-Driving Cars*.2016.[Paper](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)
