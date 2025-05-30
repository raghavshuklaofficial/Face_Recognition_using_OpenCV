---
# 🧠 **Face Recognition using OpenCV**

*Real-time face detection and recognition system leveraging OpenCV and deep learning.*

## 📋 **Overview**

This project implements a real-time face recognition system using OpenCV's deep learning module. It captures facial embeddings from images or webcam feeds and matches them against a known dataset to identify individuals.

## 🚀 **Features**

* 🎥 Real-time face detection via webcam
* 🖼️ Static image face recognition
* 🧠 Embedding extraction using deep learning models
* 🗂️ Organized face encoding storage
* 🧪 Modular and extensible codebase

## 🛠️ **Tech Stack**

| **Component**   | **Technology**       |
| --------------- | -------------------- |
| 🐍 Language     | `Python 3.x`         |
| 📷 Library      | `OpenCV`             |
| 🧠 Model        | `DNN` Module         |
| 📦 Dependencies | `NumPy`, `os`, `cv2` |

## 📁 **Project Structure**

```plaintext
face-recognition-using-openCV/
├── extract_embeddings.py           # Extracts facial embeddings from images
├── recognize_faces_in_images.py    # Recognizes faces in static images
├── recognize_faces_in_webcam.py    # Real-time face recognition via webcam
├── face_enc/                       # Directory to store face encodings
└── README.md                       # Project documentation
```

## ⚙️ **Setup & Installation**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/raghavshuklaofficial/face-recognition-using-openCV.git
   cd face-recognition-using-openCV
   ```

2. **Install dependencies:**

   ```bash
   pip install opencv-python numpy
   ```

3. **Prepare the face encodings:**

   * Place images of known individuals in the `face_enc/` directory.
   * Run the embedding extraction script:

     ```bash
     python extract_embeddings.py
     ```

4. **Run face recognition:**

   * For static images:

     ```bash
     python recognize_faces_in_images.py
     ```
   * For real-time webcam recognition:

     ```bash
     python recognize_faces_in_webcam.py
     ```

## 🧪 **Usage**

* **Adding New Faces:**

  * Add clear images of the individual's face to the `face_enc/` directory.
  * Re-run `extract_embeddings.py` to update the embeddings.

* **Recognizing Faces:**

  * Ensure the embeddings are up-to-date.
  * Use the appropriate script based on your input source (image or webcam).

---

## 👨‍💻 **Author**

**Raghav Shukla**
📌 [GitHub Profile](https://github.com/raghavshuklaofficial)

---

## 📄 **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---
## 🤝 **Contributing**

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.


Feel free to customize this `README.md` further to align with any additional features or changes in your project. Let me know if you need assistance with anything else!
