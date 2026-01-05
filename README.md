# LAOS: LLM-based Auxiliary Ophthalmic System

- Accepted to [npj Digital Medicine](https://www.nature.com/articles/s41746-025-02170-4)
## Project Background
Using "specialty voice-to-text and RAG," **LAOS** creates a full-cycle, closed loop from doctor-patient dialogue to evidence-based medical records. It also introduces a groundbreaking "clinical-semantic" dual-evaluation model, which allows the AI to understand doctors' jargon while ensuring the medical logic is perfectly sound.
The study is the first to systematically address the issue of documentation overload for ophthalmologists. It proves the superiority of AI-generated records in a clinical environment, **achieving a 62% boost in documentation speed, cutting doctors' daily overtime by an hour, and showing a significantly lower rate of critical medical errors compared to manual entry**.
<img width="1095" height="599" alt="image" src="https://github.com/user-attachments/assets/31793441-d68a-471b-b217-40275a4eeaa8" />

## Key features
1) Data-Driven, Highly Customized Architecture
<img width="1804" height="1008" alt="image" src="https://github.com/user-attachments/assets/54095c51-aa19-40d7-9abc-0b99264d0977" />
3) A Pioneering "NLP + Clinical" Dual-Evaluation Framework
<img width="1024" height="565" alt="image" src="https://github.com/user-attachments/assets/de000d4c-7648-4486-a19d-3f6cfac890b6" />


## Experimental Results
### LAOS delivers outstanding performance in both efficiency and professionalism.
- **Rapid Response**: Average speech-to-text latency is just 0.3 seconds. The system supports 30 minutes of continuous processing, with generation times far shorter than manual documentation.
- **Leading Scores**: It achieved a comprehensive clinical evaluation score of 84.1, significantly higher than the available baseline.
<img width="2404" height="699" alt="image" src="https://github.com/user-attachments/assets/6991715c-e389-4b3b-87e3-d00bf1f3cab6" />


### Scenario-Specific Performance Comparison
Discharge summaries showed the best performance due to their standardized structure. Although surgical records are more challenging (with variable procedures and frequent unexpected intraoperative events), the system still delivered statistically significant improvements in key sections like "Intraoperative Findings."
<img width="2174" height="701" alt="image" src="https://github.com/user-attachments/assets/e4e83a84-0363-48fe-9b0b-9081b6be2fd4" />


## Project layout
```
Laos_System
├── README.md                                    # Project documentation
├── requirements.txt                             # Python dependencies
├── prompt.py                                    # LLM prompts
├── utils.py                                     # Utility functions
├── main.py                                      # Launch script
├── configs/                                     # Configuration files
│ └── laos_default.yaml                          # Detail parameters
├── data/                                        # Data files
│ ├── inputs/patient_001.json                    # Patient information
│ └── refs/patient_001_admission_ref.json        # Admission Struction
└── outputs/ generated after running             # Generated results
```

## Quick start
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

## Config reference (configs/*.yaml)
- task: admission | surgery | discharge
- io:
  - patient_json: input patient info / composed JSON including transcribed speech
  - out_dir: directory for outputs
- asr: optional, enable FunASR Paraformer for speech-to-text and append to input
- search: optional, unified retrieval interface (leave base_url empty if you don’t have a service)
- doctor_feedback: optional, clinician review notes (text or file)
- llm: choose inference backend and model (suggest Qwen2-*-Instruct or Meta-Llama-3/3.1-Instruct)
- logging: log level
