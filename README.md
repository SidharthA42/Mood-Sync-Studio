# MoodSync Studio - Multimodal Emotion and Sentiment Analysis

MoodSync Studio is a Streamlit application that compares facial emotion with language sentiment to detect whether a person's visible expression and spoken or written message are aligned. It supports image upload, live webcam capture, text input, recorded audio, uploaded audio, attention visualisation, session history, and a practical action-plan view.

Live Demo : https://moodsyncstudio.streamlit.app/

The main idea is simple:

```text
Face emotion + Text or audio sentiment = Aligned or mismatch result
```

For example, if the face appears happy but the text or transcript is negative, the app can flag a mismatch instead of relying on only one input signal.

---

## Features

- Image-based facial emotion analysis using a pretrained Vision Transformer model.
- Live webcam preview using OpenCV with a clean-frame capture workflow.
- Text sentiment analysis using a pretrained RoBERTa sentiment model.
- Audio recording and audio upload using Streamlit widgets.
- Speech-to-text transcription using OpenAI Whisper.
- Analysis can run with either typed text or a transcribed audio input.
- Clear audio button for resetting recorded or uploaded audio before entering text.
- Attention heatmap for image explanation.
- Token attention chart for text explanation.
- Rule-based fusion for aligned or mismatch detection.
- Optional learned fusion using a small PyTorch fusion network.
- Session history for recent analyses.
- Action-plan tab with mismatch score, suggested next steps, reflection prompts, and downloadable report.
- Clean overview visualisation showing face input + language input = match/mismatch.

---

## System Overview

MoodSync Studio uses three main signals:

| Signal | Input | Model or Method | Output |
| --- | --- | --- | --- |
| Face | Uploaded image or captured webcam frame | `dima806/facial_emotions_image_detection` ViT model | Emotion label and confidence |
| Language | Typed text or Whisper audio transcript | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Positive, negative, or neutral sentiment |
| Fusion | Face emotion + language sentiment | Rule-based mapping or optional learned fusion | Aligned or mismatch |

The visual emotion output is mapped into sentiment polarity:

| Facial Emotion | Polarity |
| --- | --- |
| happy, surprise | positive |
| sad, angry, fear, disgust | negative |
| neutral | neutral |

If the mapped facial polarity differs from the language sentiment, the default fusion layer marks the result as `MISMATCH`. Otherwise, it marks the result as `ALIGNED`.

---

## Application Tabs

### Overview

Explains the application flow using a simple visual:

```text
Happy face in detection box + typed/spoken language = match or mismatch
```

This tab helps users understand that the project is not only detecting emotion. It compares two modalities and checks whether they agree.

### Upload Image Analysis

Allows the user to:

- Upload a face image.
- Type text, record audio, or upload audio.
- Analyse the image and language input together.
- View the detected face, predicted emotion, sentiment, fusion result, heatmap, token attention, and natural-language summary.

Text is required only when no audio transcript is available.

### Live Webcam Analysis

Allows the user to:

- Start a direct OpenCV webcam preview.
- View a live mood overlay.
- Capture a clean frame.
- Add text, recorded audio, or uploaded audio.
- Analyse the captured frame and language input together.

### Analysis History

Stores the latest upload and webcam analyses in the current Streamlit session. It shows recent emotion, sentiment, fusion result, timestamp, and summary.

### Action Plan

Converts the latest analysis into decision-support output:

- Mismatch score
- Practical response recommendations
- Reflection prompts
- Downloadable report

### Future Roadmap

Lists possible next steps such as privacy mode, model calibration, richer audio features, and deployment improvements.

---

## Project Structure

```text
MSyncStudio V1.2/
├── app.py                 # Main Streamlit UI and analysis workflow
├── audio_transcribe.py    # Whisper audio transcription helpers
├── fusion.py              # Rule-based and optional learned fusion logic
├── generator.py           # Natural-language analysis summary generator
├── image_emotion.py       # Face detection, emotion prediction, heatmap creation
├── learned_fusion.pth     # Saved weights for the optional fusion network
├── requirements.txt       # Python dependencies
├── text_sentiment.py      # RoBERTa sentiment analysis and token attention
├── utils.py               # Model loading, labels, and cached resources
└── webcam_live.py         # Alternate webcam processor retained for compatibility
```

---

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The first run downloads pretrained model weights from Hugging Face and Whisper. Use a stable internet connection.

### 3. Run the app

```bash
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## Usage Workflow

### Image Upload Workflow

1. Open `Upload Image Analysis`.
2. Upload a clear face image.
3. Provide either typed text or audio.
4. If audio is recorded or uploaded, wait for the transcript.
5. Click `Analyse`.
6. Review the face emotion, text sentiment, fusion verdict, visual explanations, and summary.

### Webcam Workflow

1. Open `Live Webcam Analysis`.
2. Click `Start Camera`.
3. Wait until the live preview detects a face.
4. Click `Capture Clean Frame`.
5. Provide either typed text or audio.
6. Click `Analyse`.
7. Review the multimodal result.

### Audio Reset Workflow

If you recorded or uploaded audio and then want to type text instead:

1. Click `Clear audio`.
2. The recorded/uploaded audio widget resets.
3. Type the text.
4. Run analysis normally.

---

## Model Details

### Facial Emotion Model

The application uses `dima806/facial_emotions_image_detection`, a Hugging Face image-classification model based on Vision Transformer architecture. It predicts seven facial emotion categories:

- angry
- disgust
- fear
- happy
- sad
- surprise
- neutral

### Text Sentiment Model

The application uses `cardiffnlp/twitter-roberta-base-sentiment-latest`, a RoBERTa-based sentiment model for English text. It predicts:

- negative
- neutral
- positive

### Audio Model

The application uses OpenAI Whisper `base` for speech-to-text transcription. The transcript is then passed into the same RoBERTa sentiment model used for typed text.

### Fusion Logic

The default fusion approach is rule-based:

```python
mapped_visual = emotion_to_sentiment[facial_emotion]
result = "MISMATCH" if mapped_visual != text_sentiment else "ALIGNED"
```

The optional learned fusion mode uses a small PyTorch neural network that receives seven image probabilities and three text probabilities as a 10-dimensional input.

---

## Explainability

MoodSync Studio includes two explanation features:

- Image heatmap: generated from ViT attention rollout and overlaid on the input image.
- Token attention: generated from the final-layer RoBERTa attention weights and displayed as a token chart.

These visualisations help users understand which parts of the image and which words influenced the model outputs.

---

## Design Decisions

### Why pretrained models were used

The project uses pretrained models because training high-quality facial emotion, speech, and transformer sentiment models from scratch requires large datasets, long training time, and GPU-level compute. During development, custom training with different emotion datasets was explored, but local CPU training was too slow and resource-heavy for reliable iteration. Pretrained models made the application practical to run on a normal laptop while still demonstrating multimodal AI concepts.

### Why Streamlit was used

Streamlit was selected because it allows fast development of an interactive machine-learning application with file uploads, audio input, webcam-related workflows, charts, and downloadable reports.

### Why fusion is kept simple

The default fusion layer is intentionally explainable. A rule-based mismatch decision is easy to inspect and suitable for a demonstration project. The optional learned fusion network is included to show how a trainable fusion layer can be added later.

---

## Limitations

- Facial emotion recognition can be affected by lighting, pose, camera quality, occlusion, and demographic bias.
- Text sentiment can miss sarcasm, cultural context, or mixed emotions.
- Whisper transcription quality depends on microphone quality, accent, noise, and speech clarity.
- The fusion result should be treated as decision support, not as a psychological diagnosis.
- The optional learned fusion layer is a lightweight demo component, not a clinically validated model.

---

## Ethical Considerations

Emotion analysis can be sensitive. This project should not be used for hiring, grading, surveillance, medical diagnosis, or automated decision-making. Users should be informed when analysis is being performed. Outputs should be interpreted carefully and combined with human judgement.

---

## Future Work

- Add stronger audio emotion features such as pitch, tone, speech rate, and energy.
- Calibrate confidence scores using a validation dataset.
- Train a real multimodal fusion model on aligned image, text, and audio samples.
- Add multilingual text and audio support.
- Add local privacy mode with explicit data retention controls.
- Add a formal evaluation suite with labelled multimodal test examples.
- Improve bias testing across age, gender expression, skin tone, lighting, and camera conditions.

---

## References

- Hugging Face model card: `dima806/facial_emotions_image_detection`
- Hugging Face model card: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- OpenAI Whisper: Robust Speech Recognition via Large-Scale Weak Supervision
- Vision Transformer: An Image is Worth 16x16 Words
- RoBERTa: A Robustly Optimized BERT Pretraining Approach
- OpenCV Cascade Classifier documentation
- Streamlit `st.audio_input` documentation
