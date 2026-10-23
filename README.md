# 🎓 SnapClass — Intelligent AI Attendance System

<p align="center">
  <strong>Automating Classroom Attendance with Face Recognition, Voice Biometrics & Machine Learning</strong>
</p>

<p align="center">
  <a href="https://snapclass-ai-system.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo"/>
  </a>
  <a href="https://github.com/Shreyas19218/Snapclass-Intelligent-AI-Attendance-System">
    <img src="https://img.shields.io/badge/Source%20Code-GitHub-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-SVM-orange?style=for-the-badge" alt="SVM"/>
</p>

## 🚀 Overview

**SnapClass** is an AI-powered attendance management application that streamlines classroom attendance through facial recognition, voice-based identification, and QR-code-powered class enrollment.

Traditional attendance systems often require manual roll calls, consume valuable classroom time, and create administrative overhead. SnapClass addresses these challenges by integrating computer vision, speaker embeddings, machine learning, and cloud-based data management into a unified Streamlit application.

The system provides dedicated teacher and student workflows for subject management, enrollment, attendance processing, and attendance record access.

🌐 **Live Demo:** [Launch SnapClass](https://snapclass-ai-system.streamlit.app/)

## ✨ Key Features

### 👨‍🏫 Teacher Portal

- Teacher registration and login.
- Create and manage subjects using subject codes.
- Share class enrollment links and dynamically generated QR codes.
- Process classroom images for face-based attendance.
- Process classroom audio for voice-based attendance.
- Review attendance results and access attendance records.
- View subject-level student and attendance information.

### 👨‍🎓 Student Portal

- Student registration and biometric enrollment.
- Face-based student identification.
- Optional voice enrollment.
- Join subjects using a subject code or QR code.
- Access enrolled subjects.
- View available attendance information.
- Unenroll from subjects where supported.

### 🤖 AI and Machine Learning

- **Facial feature extraction:** Uses Dlib's pretrained face-recognition model to generate 128-dimensional face embeddings.
- **SVM-based classification:** Uses a Support Vector Classifier to predict student identities from facial embeddings.
- **Distance-based verification:** Applies a Euclidean-distance threshold to filter face matches.
- **Audio preprocessing:** Loads and resamples audio using Librosa, then applies voice preprocessing.
- **Speaker embeddings:** Uses Resemblyzer's pretrained voice encoder to represent speaker characteristics.
- **Similarity-based identification:** Compares voice embeddings against enrolled speaker profiles.
- **Silence-based segmentation:** Splits bulk audio into voiced segments before speaker identification.

### 🔗 QR-Based Enrollment

Teachers can generate shareable subject links and QR codes using Segno. The application reads the `join-code` query parameter and routes eligible users into the student enrollment workflow.

### ☁️ Backend and Data Management

- Supabase integration for database operations.
- Separate records for teachers, students, subjects, enrollments, and attendance logs.
- Password hashing and verification using bcrypt.
- Streamlit session-state-based application routing.
- Cached model loading to reduce repeated initialization overhead.

## 🧠 AI/ML Architecture

### 1. Face Recognition Pipeline

```text
Classroom Image
      │
      ▼
Dlib Frontal Face Detector
      │
      ▼
Face Landmark Detection
      │
      ▼
Pretrained Face Recognition Model
      │
      ▼
128-Dimensional Face Embedding
      │
      ▼
SVM Classifier
      │
      ▼
Euclidean-Distance Verification
      │
      ▼
Recognized Student ID
      │
      ▼
Attendance Workflow
```

**Implementation details**

The face pipeline uses Dlib to detect faces, estimate facial landmarks, and generate a 128-dimensional descriptor for each detected face. The embeddings stored for enrolled students are used to train a scikit-learn `SVC` classifier.

The current classifier configuration is:

```python
SVC(
    kernel="linear",
    probability=True,
    class_weight="balanced"
)
```

| Parameter | Configured value | Purpose |
|---|---|---|
| `kernel` | `"linear"` | Uses a linear decision boundary in embedding space. |
| `probability` | `True` | Enables probability estimates from the classifier. |
| `class_weight` | `"balanced"` | Adjusts class weights inversely according to class frequencies. |
| Embedding dimension | 128 | Number of features in each face descriptor. |
| Face detector | Dlib frontal face detector | Detects faces in input images. |
| Landmark predictor | Pretrained Dlib shape predictor | Estimates facial landmarks for descriptor generation. |
| Verification threshold | `0.6` | Accepts a face match when the calculated Euclidean distance is at most 0.6. |

**Recognition logic**

1. Detect faces in the input image.
2. Generate a 128-dimensional embedding for each detected face.
3. Predict the student ID using the trained SVM when multiple student classes are available.
4. Compare the embedding against the stored embedding selected by the current implementation.
5. Accept the match when the Euclidean distance is less than or equal to `0.6`.

The implementation caches the Dlib models and trained classifier using Streamlit's `st.cache_resource`.

> **Technical note:** The current verification step compares against a stored embedding selected by student ID. The threshold is a configurable implementation parameter, not a measured accuracy guarantee. Recognition performance should be evaluated using representative classroom images.

### 2. Voice Recognition Pipeline

```text
Classroom Audio
      │
      ▼
Librosa Audio Loading
      │
      ▼
Resampling to 16,000 Hz
      │
      ▼
Silence-Based Audio Segmentation
      │
      ▼
Resemblyzer Voice Encoder
      │
      ▼
Speaker Embeddings
      │
      ▼
Similarity Comparison
      │
      ▼
Speaker Identification
      │
      ▼
Attendance Results
```

**Libraries and responsibilities**

- **Librosa:** Loads audio, resamples it, and segments bulk audio using silence-based splitting.
- **Resemblyzer:** Converts processed speech into speaker embeddings.
- **NumPy:** Computes similarity scores between embeddings.
- **Streamlit:** Presents processing errors and integrates recognition into the application.

**Audio processing parameters**

| Parameter | Configured value | Purpose |
|---|---|---|
| Sample rate | `16,000 Hz` | Standardizes audio input for the voice encoder. |
| Silence splitting threshold | `top_db=30` | Controls the relative amplitude threshold used to split audio around quieter sections. |
| Minimum segment duration | `0.5 seconds` | Skips segments shorter than half a second. |
| Speaker similarity threshold | `0.65` | Accepts the best speaker match when its calculated score is at least 0.65. |
| Voice encoder | `VoiceEncoder()` | Generates speaker representations using Resemblyzer. |
| Similarity calculation | `np.dot()` | Computes the dot product between the new embedding and stored embeddings. |
| Model caching | `st.cache_resource` | Reuses the initialized voice encoder across reruns. |

**Recognition logic**

1. Load audio using Librosa at a sample rate of 16 kHz.
2. For bulk attendance, split the audio into segments using `librosa.effects.split()`.
3. Ignore segments shorter than 0.5 seconds.
4. Apply Resemblyzer preprocessing and generate a speaker embedding.
5. Compare the new embedding against the stored voice embeddings.
6. Accept the best match if the score reaches the threshold of `0.65`.
7. Retain the highest matching score for each identified student in the bulk-processing results.

> **Technical note:** The current implementation uses a dot product as its similarity score. It does not explicitly normalize both embeddings before this comparison in the visible pipeline code. Therefore, `0.65` should not automatically be interpreted as a calibrated probability or guaranteed accuracy level.

## 🛠️ Technology Stack

| Technology / Library | Application |
|---|---|
| Python | Core application logic |
| Streamlit | Web interface, dialogs, session state, caching, and deployment |
| Dlib | Face detection, facial landmark prediction, and face descriptors |
| `face_recognition_models` | Pretrained Dlib model assets |
| Scikit-learn | SVM classification |
| NumPy | Embedding representation and numerical similarity calculations |
| Librosa | Audio loading, resampling, and segmentation |
| Resemblyzer | Voice embeddings and speaker identification |
| Pillow | Image-related utilities |
| Pandas | Tabular data manipulation |
| Segno | QR-code generation |
| Supabase Python SDK | Database connectivity and CRUD operations |
| bcrypt | Password hashing and verification |
| Git and GitHub | Version control and source hosting |
| Streamlit Community Cloud | Application deployment |

## 🏗️ Project Structure

```text
Snapclass-Intelligent-AI-Attendance-System/
│
├── app.py
│
├── requirements.txt
│
├── .gitignore
│
└── src/
    │
    ├── components/
    │   ├── dialog_auto_enroll.py
    │   ├── dialog_voice_attendance.py
    │   ├── dialog_add_photo.py
    │   ├── dialog_attendance_results.py
    │   ├── dialog_enroll.py
    │   ├── dialogue_create_subject.py
    │   ├── dialogue_share_subject.py
    │   ├── subject_card.py
    │   ├── header.py
    │   └── footer.py
    │
    ├── database/
    │   ├── config.py
    │   └── db.py
    │
    ├── pipelines/
    │   ├── face_pipeline.py
    │   └── voice_pipeline.py
    │
    ├── screens/
    │   ├── home_screen.py
    │   ├── teacher_screen.py
    │   └── student_screen.py
    │
    └── ui/
        └── base_layout.py
```


### Folder Responsibilities

**`app.py` — Application entry point**

- Configures the Streamlit page.
- Initializes the login state.
- Routes users to the home, teacher, or student screen.
- Reads the `join-code` URL query parameter.
- Triggers the automatic enrollment dialog for eligible student sessions.

**`src/screens/` — Main application screens**

- `home_screen.py`: Entry point for the teacher and student workflows.
- `teacher_screen.py`: Teacher-facing subject management and attendance workflows.
- `student_screen.py`: Student-facing registration, enrollment, and attendance features.

**`src/components/` — Reusable UI components**

- Enrollment and automatic enrollment dialogs.
- Voice attendance and classroom image dialogs.
- Attendance result dialogs.
- Subject creation and sharing dialogs.
- Subject cards, headers, and footers.

**`src/pipelines/` — AI and ML processing**

- `face_pipeline.py`: Face detection, 128-dimensional embedding generation, SVM training, and face-match verification.
- `voice_pipeline.py`: Audio preprocessing, speaker embedding generation, silence-based segmentation, and speaker matching.

**`src/database/` — Database integration**

- `config.py`: Initializes the Supabase client using Streamlit Secrets.
- `db.py`: Implements database operations for authentication, students, subjects, enrollment, and attendance.

**`src/ui/` — Interface styling**

- `base_layout.py`: Shared UI layout and styling definitions.

**`requirements.txt` — Dependencies**

Lists the Python packages needed to run the application, including computer vision, audio processing, machine learning, database connectivity, and QR-code generation libraries.

## 🗄️ Database Design

The application interacts with the following Supabase tables in the current database module:

| Table | Purpose |
|---|---|
| `teachers` | Teacher accounts and password hashes |
| `students` | Student profiles and stored biometric embeddings |
| `subjects` | Subject details, codes, sections, and teacher associations |
| `subject_students` | Student-to-subject enrollment relationships |
| `attendance_logs` | Attendance records associated with students and subjects |

The database module provides operations for account creation and login, retrieving students, creating subjects, managing enrollments, and recording attendance.

**Security consideration:** Biometric embeddings are sensitive data. Apply appropriate database access policies, limit access to authorized users, and avoid exposing student information or secrets in public repositories.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.12 as a starting point, subject to dependency compatibility.
- Git installed on your machine.
- A Supabase project with the required schema and access policies.
- A webcam or classroom images for face attendance.
- Audio input or recordings for voice attendance.

### 1. Clone the repository

```bash
git clone https://github.com/Shreyas19218/Snapclass-Intelligent-AI-Attendance-System.git
cd Snapclass-Intelligent-AI-Attendance-System
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some computer-vision and audio dependencies may require additional platform-specific configuration.

### 4. Configure Supabase

Create a local `.streamlit/secrets.toml` file:

```toml
SUPABASE_URL = "your-supabase-project-url"
SUPABASE_KEY = "your-supabase-key"
```

Replace the placeholders with your actual credentials. The key names must match `src/database/config.py`.

**Never commit actual credentials to GitHub.** Configure secrets separately for Streamlit Community Cloud.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit in your browser.

## ☁️ Deployment

SnapClass is deployed through Streamlit Community Cloud.

To deploy your own instance:

1. Push the repository to GitHub.
2. Connect the repository to Streamlit Community Cloud.
3. Select the appropriate branch and `app.py` as the main file.
4. Choose a Python version compatible with all dependencies.
5. Configure `SUPABASE_URL` and `SUPABASE_KEY` in the app's Secrets settings.
6. Deploy and verify the teacher, student, face, voice, and QR enrollment workflows.

For future updates, push your changes to the connected GitHub branch and verify the deployment status and logs.

## 📊 Model Evaluation and Limitations

The project implements face classification and speaker matching, but the code configuration alone does not establish real-world recognition accuracy.

For a more complete evaluation, measure:

- Face identification precision, recall, and false-match rate.
- Voice identification precision, recall, and false-acceptance rate.
- Recognition performance under varying lighting and noise conditions.
- End-to-end attendance processing time.
- Performance with different numbers of enrolled students.

Use a held-out evaluation set and document the testing methodology before publishing numerical performance claims.

## 🔒 Security and Privacy

- Store secrets outside source control.
- Hash passwords using bcrypt rather than storing plaintext passwords.
- Restrict Supabase database access with appropriate authorization policies.
- Protect stored face and voice embeddings as sensitive biometric information.
- Obtain consent before processing biometric data.
- Validate identity matches and attendance results before relying on them for consequential decisions.

## 🚀 Future Enhancements

- Automated attendance analytics and downloadable reports.
- Improved audio segmentation in noisy classroom environments.
- Formal evaluation and calibration of face and voice matching thresholds.
- Better handling of unknown faces and speakers.
- More comprehensive automated tests and error reporting.
- Stronger database authorization and biometric-data retention controls.

## 💡 Key Engineering Highlights

- Integrated computer vision and speaker recognition into a single application.
- Implemented SVM-based student classification from facial embeddings.
- Combined pretrained biometric representations with threshold-based verification.
- Built audio preprocessing and speaker matching workflows.
- Developed QR-based enrollment through URL query parameters.
- Integrated cloud database operations with a modular Streamlit application.
- Applied model caching to reduce repeated initialization.

## 👨‍💻 Author

**Shreyas Snehal**

B.Tech — Computer Science and Engineering  
Birla Institute of Technology, Mesra

- **GitHub:** [Shreyas19218](https://github.com/Shreyas19218)
- **Project Repository:** [SnapClass — Intelligent AI Attendance System](https://github.com/Shreyas19218/Snapclass-Intelligent-AI-Attendance-System)
- **Live Application:** [SnapClass on Streamlit](https://snapclass-ai-system.streamlit.app/)

<p align="center">
  <strong>SnapClass — Making Classroom Attendance Smarter with AI.</strong>
</p>
