# LAOS: Structured Documentation for Ophthalmology Day Surgery

This repository provides a configurable end-to-end pipeline that converts clinical speech/text notes into structured JSON documents for:
- Admission (admission)
- Surgery (surgery)
- Discharge summary (discharge)

Key features
- Config-driven: only keep the `--config` CLI flag; all details are specified in `configs/*.yaml`.
- ASR support: FunASR Paraformer (used in the paper) can be enabled with one switch; falls back to Whisper or SpeechRecognition.
- LLM support: Qwen and LLaMA families (Transformers CausalLM with automatic chat-template application). Also supports OpenAI or HF pipeline.

Project layout
- README.md
- requirements.txt
- prompt.py
- utils.py
- main.py
- configs/
  - laos_default.yaml
- data/
  - kb/kb_sample.txt
  - inputs/patient_001.json
  - refs/patient_001_admission_ref.json
- outputs/ generated after running

Quick start
1) Install dependencies
   - Python 3.10+ recommended
   - For GPU: install a PyTorch build that matches your environment (see “Dependency installation tips”)
   - Other dependencies: `pip install -r requirements.txt`

2) Run a sample (for Qwen/LLaMA, ensure you can pull from Hugging Face or have local weights)
   - Edit paths in `configs/laos_default.yaml` (e.g., patient_json / audio_path)
   - Run:
     ```
     python main.py --config configs/laos_default.yaml
     ```
   - Outputs will be written to `outputs/`, for example:
     ```
     outputs/patient_001_admission_pred.json
     ```

Config reference (configs/*.yaml)
- task: admission | surgery | discharge
- io:
  - patient_json: input patient info / composed JSON including transcribed speech
  - out_dir: directory for outputs
- asr: optional, enable FunASR Paraformer for speech-to-text and append to input
- search: optional, unified retrieval interface (leave base_url empty if you don’t have a service)
- doctor_feedback: optional, clinician review notes (text or file)
- llm: choose inference backend and model (suggest Qwen2-*-Instruct or Meta-Llama-3/3.1-Instruct)
- logging: log level
