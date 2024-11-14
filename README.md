# WEAL - Wellness Enhancement and Awareness Lifeline

## Overview
WEAL is a comprehensive healthcare platform combining preventive care, mental health support, treatment facilitation, and advanced diagnostic capabilities through the W.I.S.E. (WEAL Integrated Scan Evaluation) system.

[download for complete files] 
https://drive.google.com/drive/folders/1rCVZNv0omCJMPEsmgq1ggl1XXENDyypI?usp=sharing 

## Directory Structure
```
├── app.py                     # Main Flask application
├── WISE/                      # Brain tumor detection system
│   ├── app.py                # WISE module application
│   ├── classify.py           # Classification logic
│   ├── resnet.py            # ResNet implementation
│   ├── static/              # Static files for WISE
│   └── templates/           # WISE HTML templates
├── templates/                # Main application templates
├── static/                  # Static assets
├── assets/                 # Core assets
│   ├── css/               # Stylesheet files
│   ├── js/               # JavaScript files
│   ├── scss/             # SASS source files
│   └── imgs/             # Image assets
└── vendors/              # Third-party libraries
```

## Features

### 1. PREVENT
- Yoga and meditation guidance
- Diet planning and nutrition advice
- Expert consultation services
- Progress tracking

### 2. MENTAL HEALTH
- OpenAI-powered chatbot support
- 24/7 mental health assistance
- Resource recommendations

### 3. TREAT
- Symptom-based doctor recommendations
- Focus on government healthcare professionals
- Affordable consultation platform
- Specialist referral system

### 4. W.I.S.E. System
- Brain tumor detection and classification
- Multiple model comparison:
  - ResNet with U-Net filtering
  - VGG16 classification
  - Comparative models (AlexNet, InceptionV3, LeNet)

## Installation

### 1. Basic Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/weal-healthcare.git

# Install dependencies
flask==2.0.1
openai==0.27.0
tensorflow==2.8.0
keras==2.8.0
scikit-learn==0.24.2
numpy==1.19.5
pandas==1.3.3
opencv-python==4.5.3.56
pillow==8.3.2
python-dotenv==0.19.0
joblib==1.0.1

# Set up environment variables
cp .env.example .env
# Edit .env with your configurations

# Run the application
python app.py
```

### 2. Required Downloads
Due to size limitations, download the following from [Google Drive Link] and place in appropriate directories:

**WISE Models:**
- `WISE/AlexNet_best_model.h5`
- `WISE/VGG16_best_model.h5`
- `WISE/InceptionV3_best_model.h5`
- `WISE/LeNet_best_model.h5`
- `WISE/Dilated_CNN_best_model.h5`
- `WISE/best_model.h5`
- `WISE/final_xception.h5`
- `WISE/vgg16-improved-weights.h5`

**Media Files:**
- Video assets in `static/` and `templates/`
- Large image files in `assets/imgs/`

## Files on GitHub
### Core Application Files:
- `app.py`
- `mental.html`
- `Diet.html`
- `Treat.html`
- `Yoga.html`
- `index.html`
- `login.html`
- `signup.html`

### WISE System:
- `WISE/app.py`
- `WISE/classify.py`
- `WISE/resnet.py`
- `WISE/templates/*`

### Data Files:
- `disease_description.csv`
- `disease_precaution.csv`
- `Doctor_Versus_Disease.csv`
- `trained_model.joblib`
- `vectorizer.joblib`

### Assets:
- CSS files
- JavaScript files
- SCSS files
- Small images and icons

## Tech Stack
- **Frontend**: HTML5, SCSS, Bootstrap 4.3.1
- **Backend**: Python Flask
- **ML/DL**: TensorFlow, Keras, scikit-learn
- **AI Integration**: OpenAI API
- **Asset Processing**: jQuery, themify-icons

## Development
```bash
# Install SCSS compiler
npm install -g sass

# Compile SCSS
sass assets/scss/creative-studio.scss assets/css/creative-studio.css
```

## License
MIT
