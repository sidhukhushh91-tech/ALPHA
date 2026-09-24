# Product Requirements Document (PRD)

## 4.1 Document Information
- **Project name:** Sound-Based Machine Health Monitor
- **Document title:** Product Requirements Document
- **Version:** 1.0
- **Status:** Draft
- **Date:** [To be filled at submission]
- **Intended audience:** Project supervisor, evaluators, development team (student author)

---

## 4.2 Executive Summary
The Sound-Based Machine Health Monitor is a prototype application that uses audio signal processing and machine learning to classify machine operating sounds as normal or abnormal. A user uploads or records a short audio clip; the system extracts acoustic features, runs a trained classifier, and returns a result with a confidence indicator on a simple dashboard. The project demonstrates an end-to-end applied ML pipeline — data preprocessing, feature engineering, model training/evaluation, and backend/frontend integration — applied to a genuine industrial maintenance problem (acoustic-based predictive maintenance), scoped to be realistically achievable with student-level resources.

---

## 4.3 Background and Problem Statement
Unplanned machine downtime is expensive, and both purely manual inspection and fixed-interval maintenance have well-known drawbacks:
- **Manual inspection** depends on the availability and experience of technicians and cannot run continuously.
- **Time-based maintenance** either wastes resources servicing healthy machines or misses faults that develop faster than the inspection schedule.
- Many machines emit distinguishable sounds during normal operation, and deviations in these sounds (pitch changes, added harmonics, impulsive noises) can be an early, low-cost signal of a developing issue — this is the premise behind real-world "acoustic condition monitoring" used in industry.

This project does **not** claim sound alone can diagnose all faults with certainty; it demonstrates a realistic, affordable software prototype that shows the technique's value as a supporting signal, buildable with open datasets, a laptop, and standard Python ML/audio libraries.

---

## 4.4 Product Vision
Make basic machine-sound analysis accessible to anyone with a laptop and a microphone or a set of recordings — lowering the barrier to acoustic condition monitoring so it is not limited to expensive proprietary industrial systems, starting with a clear, explainable prototype that can grow toward more advanced monitoring over time.

---

## 4.5 Goals and Objectives

**Primary objectives**
- Build a working prototype that classifies uploaded/recorded machine audio as normal, abnormal, or uncertain.
- Provide a usable dashboard that displays the result with a confidence value.

**Secondary objectives**
- Visualize the audio (waveform/spectrogram) to make the analysis interpretable.
- Maintain a simple history of past analyses.

**Technical objectives**
- Implement a complete audio ML pipeline: validation → preprocessing → feature extraction → inference → structured output.
- Achieve a documented, evaluated (not merely claimed) classification performance on a held-out test set.
- Expose the functionality through a REST API with clear request/response contracts.

**Academic learning objectives**
- Demonstrate applied understanding of audio signal processing (MFCCs, spectrograms).
- Demonstrate model selection, training, and evaluation methodology (train/validation/test discipline, avoiding data leakage).
- Demonstrate full-stack integration (ML model → backend API → frontend → database).

---

## 4.6 Target Users and Personas

**Persona 1 — Factory Maintenance Technician ("Raj")**
- Role: Performs routine machine checks on a factory floor.
- Main needs: A quick way to flag machines that need closer inspection sooner rather than later.
- Challenges: Limited time to inspect every machine manually every day.
- Expected benefit: A lightweight tool that surfaces likely problem machines from a quick audio check.
- Typical interaction: Records 5–10 seconds of a machine running, uploads it, checks the result before deciding whether to schedule inspection.

**Persona 2 — Small Manufacturing Business Owner ("Simran")**
- Role: Owns a small workshop with a handful of machines and no dedicated maintenance engineer.
- Main needs: An affordable way to catch problems before a costly breakdown.
- Challenges: Cannot afford industrial IoT monitoring systems.
- Expected benefit: A free/low-cost prototype tool usable with a phone or basic microphone.
- Typical interaction: Periodically records machine sounds and checks the dashboard for anomaly flags.

**Persona 3 — University Student/Researcher ("Reviewer")**
- Role: Evaluates or extends the project academically.
- Main needs: Clear documentation, reproducible results, understandable methodology.
- Challenges: Needs to verify the technical soundness of the approach quickly.
- Expected benefit: Well-documented pipeline and evaluation metrics to review.
- Typical interaction: Reads documentation, runs the demo, inspects model evaluation results.

---

## 4.7 Product Scope

**In scope (initial working prototype)**
- Audio upload and (optionally) browser-based microphone recording
- Audio validation (format, size, duration, silence detection)
- Feature extraction (MFCCs, spectral features, and/or Mel-spectrogram generation)
- A single well-evaluated ML classifier (normal / abnormal, with an uncertainty threshold)
- REST API for submitting audio and receiving predictions
- Dashboard showing result, confidence, and a basic visualization
- Local analysis history (SQLite)

**Out of scope (initial version)**
- Real-time/continuous streaming monitoring
- IoT sensor integration
- Machine-specific fault-type diagnosis beyond normal/abnormal (unless a suitable labeled dataset is available)
- Multi-user accounts, authentication, cloud deployment
- Mobile native application

**Future enhancements**
- Continuous monitoring via connected microphones/IoT
- Multi-machine dashboard with historical trend charts
- Remaining-useful-life estimation
- Machine-specific fault categorization with richer datasets
- Edge deployment for offline inference

---

## 4.8 Functional Requirements

| ID | Name | Description | Priority | Acceptance Criteria |
|---|---|---|---|---|
| FR-001 | Audio file upload | User can upload an audio file through the dashboard | Mandatory | File is accepted, validated, and passed to preprocessing |
| FR-002 | Microphone recording | User can record audio directly in-browser | Optional | Recorded clip is captured and processed identically to an upload |
| FR-003 | Format validation | System rejects unsupported audio formats with a clear message | Mandatory | Invalid format triggers a specific error response, not a crash |
| FR-004 | Duration/size validation | System enforces max duration and file size | Mandatory | Oversized/too-long files are rejected with an explanatory message |
| FR-005 | Audio preprocessing | System resamples, trims silence, and normalizes volume | Mandatory | Preprocessed audio meets defined format before feature extraction |
| FR-006 | Feature extraction | System computes MFCCs/spectral features and/or spectrogram image | Mandatory | Feature vector/image generated for every valid clip |
| FR-007 | ML inference | System runs the trained model on extracted features | Mandatory | A prediction is returned for every valid input |
| FR-008 | Normal/abnormal classification | System outputs a class label | Mandatory | Label is one of: normal, abnormal, uncertain |
| FR-009 | Confidence display | System shows a confidence/uncertainty value with the prediction | Mandatory | Confidence value is numeric and displayed in the UI |
| FR-010 | Health status visualization | Dashboard shows a clear visual status indicator | Mandatory | Status is visually distinguishable (e.g., color-coded) |
| FR-011 | Waveform/spectrogram visualization | Dashboard displays a plot of the analyzed audio | Optional | Plot renders for a valid processed clip |
| FR-012 | Analysis history | Past analyses are listed with timestamp and result | Optional | History persists across sessions via the database |
| FR-013 | Prediction report | System can export a simple result summary | Future enhancement | N/A for initial prototype |
| FR-014 | Error messages | User-friendly messages for all failure cases | Mandatory | No unhandled exceptions reach the user |
| FR-015 | Responsive dashboard | UI adapts to common screen sizes | Optional | Layout remains usable at common laptop/tablet widths |
| FR-016 | Sample audio demos | Preloaded example clips for demonstration | Optional | At least one sample per class available for the live demo |

---

## 4.9 Non-Functional Requirements

| ID | Category | Requirement (target, not yet achieved) |
|---|---|---|
| NFR-001 | Performance | Target: end-to-end prediction returned in under 5 seconds on a standard laptop |
| NFR-002 | Reliability | System should handle invalid input without crashing |
| NFR-003 | Usability | A first-time user should be able to complete an analysis without instructions, within a few minutes |
| NFR-004 | Maintainability | Code organized into clearly separated modules (preprocessing, inference, API, UI) |
| NFR-005 | Scalability | Not a hard requirement for the prototype; architecture should not actively prevent future scaling |
| NFR-006 | Compatibility | Should run on a standard laptop (Windows/macOS/Linux) with Python 3.10+ |
| NFR-007 | Security | Uploaded files validated before processing; no arbitrary code execution from file content |
| NFR-008 | Privacy | No unnecessary personal data collected; audio retention policy clearly stated to the user |
| NFR-009 | Accessibility | Basic readable contrast and labeling in the UI |
| NFR-010 | Error recovery | A failed analysis should not corrupt stored history or crash the backend |

---

## 4.10 User Stories

- As a technician, I want to record a machine's sound directly in the app, so that I don't need a separate recording tool.
- As a technician, I want to upload an existing audio file, so that I can analyze previously recorded clips.
  - *Acceptance:* Upload completes and a result is returned, or a clear error is shown.
- As a user, I want to see a confidence score with the result, so that I know how much to trust an "abnormal" flag.
  - *Acceptance:* Every prediction includes a numeric confidence/uncertainty value.
- As a user, I want an "uncertain" outcome when the system isn't confident, so that I'm not misled by a false-certain answer.
  - *Acceptance:* Predictions below a defined confidence threshold are labeled uncertain, not forced into normal/abnormal.
- As a user, I want to review past analyses, so that I can track a machine's condition over time.
  - *Acceptance:* History list shows timestamp, filename/reference, and result for each past run.

---

## 4.11 Use Cases

**UC-01: Analyze Uploaded Audio**
- Actor: User
- Preconditions: Application running; trained model available
- Main flow: User uploads file → system validates → preprocesses → extracts features → runs inference → displays result
- Alternative flow: User cancels upload before submission
- Error conditions: Invalid format, oversized file, corrupted audio → system shows specific error
- Expected outcome: A classification result with confidence is displayed, or a clear error message

**UC-02: Record and Analyze Audio**
- Actor: User
- Preconditions: Browser microphone permission granted
- Main flow: User records clip → clip is validated and processed identically to UC-01
- Alternative flow: User denies microphone permission
- Error conditions: No microphone detected, permission denied, silent recording
- Expected outcome: Same as UC-01, or a permission/silence-specific error

**UC-03: View Analysis History**
- Actor: User
- Preconditions: At least one prior analysis stored
- Main flow: User opens history panel → system retrieves stored records → displays list
- Error conditions: Database unavailable
- Expected outcome: List of past analyses, or a graceful "history unavailable" message

---

## 4.12 Complete System Workflow
Machine Sound → Audio Input (upload/record) → Validation (format, size, duration) → Preprocessing (resample, trim silence, normalize) → Feature Extraction (MFCCs / spectral features / Mel-spectrogram) → ML Model → Classification (normal/abnormal) → Confidence/Uncertainty Assessment (threshold check) → Dashboard Result (status + confidence + visualization) → Optional History Storage (SQLite).

Each stage validates its own input and fails gracefully with a specific, user-readable error rather than propagating an unhandled exception to the next stage.

---

## 4.13 AI/ML Requirements

- **Problem type:** Supervised classification (normal vs. abnormal machine sound), with a confidence threshold used to output "uncertain" for low-confidence predictions.
- **Input data:** Short audio clips (a few seconds) per machine/class.
- **Expected output:** Class label + confidence score.
- **Feature approach considered:**
  - *Traditional ML on extracted features* (MFCCs, spectral centroid, zero-crossing rate, RMS energy, etc.) fed into a classifier such as Random Forest or an MLP (ANN).
  - *CNN on Mel-spectrogram images*, treating the spectrogram as an image classification problem.
  - *Anomaly detection* (e.g., one-class models) if abnormal samples are too scarce for balanced supervised classification.
- **Recommended initial approach:** Start with a traditional ML classifier on extracted numeric features (fast to train, easier to evaluate and explain in a viva) as the baseline; add a CNN-on-spectrogram model as a comparison if time and data permit. One well-evaluated model is sufficient for the prototype; a second model is a valuable comparison, not a requirement.
- **Validation/testing:** Proper train/validation/test split, stratified by class; no clips from the same recording session should span both train and test sets (to avoid data leakage).
- **Handling unfamiliar/low-quality input:** Low-confidence predictions are labeled "uncertain" rather than forced into a class; very low-energy (near-silent) clips are rejected at the validation stage rather than passed to the model.
- **Known limitations:** Performance is bounded by dataset size/diversity; the model will generalize best to machine types and fault types represented in training data, and may perform poorly on unfamiliar machines.

---

## 4.14 Dataset Requirements
- **Selection criteria:** Labeled machine-operation audio with at least "normal" and "abnormal" categories.
- **Candidate source:** Publicly available industrial sound datasets (e.g., MIMII) — license terms and suitability must be verified before use and cited in the report.
- **Labeling:** Use the dataset's existing labels; do not relabel without justification.
- **Cleaning/preprocessing:** Remove corrupted/empty files; standardize sample rate and clip length.
- **Splits:** Stratified train/validation/test split (e.g., 70/15/15), documented in the report with exact counts once known.
- **Class imbalance:** Expect abnormal samples to be rarer than normal; document the actual class distribution and account for it in evaluation (e.g., report precision/recall per class, not just accuracy).
- **Licensing/provenance:** Cite the dataset source and license explicitly in the final report and README.
- **Note:** Machine-specific fault classification (beyond normal/abnormal) requires a dataset with machine-specific labels — only attempt this if such data is actually available.

---

## 4.15 Frontend Requirements
- Home/overview page with a short project description
- Audio upload area (drag-and-drop or file picker)
- Optional recording interface with a visible recording indicator
- Analysis progress indicator (spinner/progress bar) during processing
- Results panel: predicted class, confidence value, and status
- Health status indicator (clear visual distinction, e.g., color-coded badge)
- Waveform or spectrogram visualization of the analyzed clip
- Analysis history panel/table
- Clear, non-technical error messages
- Responsive layout for common laptop screen widths

Design should remain simple and uncluttered — a small number of clearly labeled panels rather than a dense dashboard.

---

## 4.16 Backend Requirements
- Expose REST endpoints for: audio submission/analysis, history retrieval, and (optionally) health check.
- Validate all incoming files before passing them to preprocessing.
- Integrate the preprocessing pipeline and model inference as internal service calls (not duplicated logic).
- Format model output into a consistent JSON response schema.
- Persist each analysis result to the database.
- Handle and log errors at each pipeline stage with enough detail for debugging, without exposing internal stack traces to the client.

---

## 4.17 Database Requirements
Proposed table: `analyses`
| Field | Type | Notes |
|---|---|---|
| analysis_id | INTEGER (PK) | Auto-increment |
| timestamp | DATETIME | When the analysis was run |
| machine_category | TEXT (nullable) | If the user optionally tags a machine type |
| audio_reference | TEXT | Filename or generated reference (not necessarily the raw audio itself) |
| predicted_class | TEXT | normal / abnormal / uncertain |
| confidence | FLOAT | Model confidence/uncertainty value |
| processing_status | TEXT | success / error |
| result_summary | TEXT | Optional short human-readable summary |

Raw audio files should be treated as **temporary** unless the user is explicitly informed that recordings are retained; only the structured analysis record needs to persist. No unnecessary personal data (names, precise location, etc.) should be collected.

---

## 4.18 System Architecture
See `docs/SYSTEM_ARCHITECTURE.md` for the full diagram and component breakdown. At a summary level: Frontend (browser) → Backend API (FastAPI) → Preprocessing/Feature Extraction module → ML Inference module (loads a pre-trained, serialized model) → Database (SQLite, for history) → Response back to Frontend.

---

## 4.19 Error Handling and Edge Cases
| Condition | Expected System Response |
|---|---|
| Unsupported file format | Reject with a specific "unsupported format" message |
| Empty audio file | Reject at validation, before preprocessing |
| Corrupted audio | Catch decode error, return a clear error message |
| Excessively long recording | Reject or truncate per configured max duration, with a message |
| Invalid microphone permissions | Frontend shows a permission-request message; no crash |
| Background noise | Preprocessing attempts normalization; prediction confidence may drop, which is surfaced honestly rather than hidden |
| Silent audio | Detected at validation (near-zero energy) and rejected with a message |
| Missing model file | Backend returns a service-unavailable error, not a silent failure |
| Model inference failure | Caught and logged; user sees a generic "analysis failed, please try again" message |
| Database failure | Analysis result still returned to the user; history logging failure is logged separately, not shown as an analysis failure |
| Unsupported machine category | If category tagging is used, an unrecognized tag is stored as "unspecified" rather than rejected |
| Uncertain predictions | Explicitly labeled "uncertain" rather than defaulted to normal or abnormal |

---

## 4.20 Security and Privacy
- Validate file type and size before any processing (avoid passing untrusted files to parsing libraries unchecked).
- Store uploaded audio only temporarily unless the user is clearly informed otherwise; delete temp files after processing.
- Restrict access to any stored recordings/history to the local application (no public endpoint exposing raw audio).
- Define and communicate a simple data retention policy (e.g., "audio is processed and discarded; only the classification result is stored").
- Avoid collecting any personal information beyond what is functionally necessary (no names, no precise location by default).

---

## 4.21 Constraints and Assumptions
- Development resources are limited to student-level hardware (a standard laptop, no dedicated GPU assumed).
- Dataset availability is limited to public sources; results are bounded by what those datasets contain.
- Microphone quality and ambient noise are uncontrolled variables during live demos.
- Computational budget assumes CPU-only inference for the deployed prototype.
- Browser compatibility assumed for modern Chrome/Edge/Firefox versions (for the recording feature specifically).
- The model's generalization is limited to machine types/fault types represented in the training data.
- Access to real industrial machines for original data collection is not assumed; public datasets are the primary source.

---

## 4.22 Risks and Mitigation

| Risk ID | Description | Likelihood | Impact | Mitigation | Contingency |
|---|---|---|---|---|---|
| R-01 | Insufficient labeled data for reliable training | Medium | High | Use an established public dataset; augment data where feasible | Narrow scope to normal/abnormal only, on the best-supported machine type |
| R-02 | Class imbalance skews accuracy | Medium | Medium | Evaluate with precision/recall/F1 per class, not accuracy alone | Apply class weighting or resampling |
| R-03 | Overfitting on a small dataset | Medium | Medium | Use proper train/val/test splits, cross-validation, regularization | Simplify model, add more data via augmentation |
| R-04 | Poor-quality live demo recordings (noise) | Medium | Medium | Provide preloaded sample clips as a fallback for the live demo | Rely on sample clips if a live mic recording fails |
| R-05 | Model performs poorly on unfamiliar machine sounds | High | Medium | Clearly scope and document supported machine types | Present as a documented limitation, not a failure |
| R-06 | Time constraints limit full feature set delivery | Medium | Medium | Prioritize mandatory FRs first (see Development Plan) | Cut optional features (history, waveform viz) if needed |

---

## 4.23 Testing and Acceptance Criteria
Full detail in `docs/TESTING_PLAN.md`. Summary criteria:
- Valid audio input produces a classification with confidence in under the target response time.
- Invalid inputs (bad format, empty, oversized, corrupted, silent) are rejected with specific, correct error messages — never an unhandled crash.
- Model evaluation reports accuracy, precision, recall, F1-score, and a confusion matrix on a held-out test set.
- API endpoints return correctly structured JSON for both success and error cases.
- Dashboard displays results correctly for at least one clip per class during manual UI testing.
- Database correctly stores and retrieves history records.

---

## 4.24 Success Metrics
- Classification accuracy on held-out test data (reported, not assumed)
- Precision, recall, and F1-score per class
- Confusion matrix
- False-positive rate and false-negative rate (false negatives — missed abnormal sounds — are the more costly error class and should be highlighted separately)
- Inference time per clip
- Successful audio-processing rate (percentage of valid uploads that complete without error)
- Qualitative: whether a first-time user can complete an analysis unaided

**Note:** Because abnormal samples are typically the minority class, accuracy alone can be misleading (a model that always predicts "normal" can still score high accuracy). Precision/recall/F1 on the abnormal class specifically should be treated as the primary indicators of real performance.

---

## 4.25 Development Roadmap
1. **Requirement analysis** — finalize scope, confirm dataset availability. *Deliverable: this PRD.*
2. **Dataset exploration** — inspect class balance, sample quality. *Deliverable: dataset summary notes.*
3. **Data preprocessing** — build validation, resampling, normalization pipeline. *Deliverable: `preprocessing.py`.*
4. **Baseline model development** — train the initial classifier on extracted features. *Deliverable: trained baseline model + training script.*
5. **Model evaluation** — compute accuracy/precision/recall/F1/confusion matrix on test split. *Deliverable: evaluation report.*
6. **Backend development** — build FastAPI endpoints and integrate inference. *Deliverable: working API.*
7. **Frontend development** — build upload/record UI and results dashboard. *Deliverable: working UI.*
8. **Integration** — connect frontend, backend, database end-to-end. *Deliverable: integrated prototype.*
9. **Testing** — execute the testing plan, fix defects. *Deliverable: test results.*
10. **Documentation** — finalize all `docs/` files and the academic report. *Deliverable: complete documentation set.*
11. **Final demonstration** — prepare and rehearse the live demo. *Deliverable: demo script/slides.*

Each phase's completion criteria are detailed in `docs/DEVELOPMENT_PLAN.md`.

---

## 4.26 Future Enhancements *(explicitly future work, not part of the initial prototype)*
- Continuous/streaming monitoring
- IoT sensor integration for always-on listening
- Multi-machine, multi-location monitoring
- Machine-specific fault categories (given suitable data)
- Historical trend analysis and remaining-useful-life estimation
- Edge-device deployment for offline/low-connectivity use
- Integration with existing maintenance-management systems

---

## 4.27 Glossary
- **Acoustic anomaly:** A sound pattern that deviates from a machine's established normal operating sound.
- **Audio preprocessing:** Preparing raw audio (resampling, trimming, normalizing) before feature extraction.
- **MFCC (Mel-Frequency Cepstral Coefficients):** A compact numeric representation of the timbre/spectral shape of audio, widely used in audio ML.
- **Spectrogram:** A visual representation of a signal's frequency content over time.
- **Feature extraction:** Converting raw data into a set of numeric or structured inputs a model can learn from.
- **Classification:** Assigning an input to one of a fixed set of categories.
- **Anomaly detection:** Identifying inputs that deviate from a learned notion of "normal," often without needing labeled abnormal examples.
- **Precision:** Of all items predicted positive (e.g., abnormal), the proportion that were actually positive.
- **Recall:** Of all items that were actually positive, the proportion the model correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **Model inference:** Running a trained model on new input to produce a prediction.
- **Data leakage:** When information from outside the training set (e.g., overlapping samples) improperly influences model evaluation, inflating apparent performance.

---

## 4.28 Conclusion
The Sound-Based Machine Health Monitor is a scoped, realistic prototype that applies audio signal processing and supervised machine learning to a genuine maintenance problem. It is intentionally limited in scope — normal/abnormal classification on available public data, not comprehensive industrial fault diagnosis — so that it can be implemented, evaluated, and demonstrated reliably within student-project constraints, while still reflecting a real and actively used industrial technique (acoustic-based predictive maintenance). Its principal value is as a learning and demonstration vehicle for the full applied-ML pipeline, with a clear, honestly documented path toward more advanced future work.
