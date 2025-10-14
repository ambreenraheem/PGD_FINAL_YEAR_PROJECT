## NED UNIVERSITY OF ENGINEERING AND TECHNOLOGY
## Centre of Multidisciplinary Postgraduate Programs (CMPP) Postgraduate Diploma (PGD) Programs
#### INTERIM PROJECT REPORT

### Student Name: Ambreen Abdul Raheem
### Supervisor Name: Sir Imran Bashir
### Program Name: PGD in Data Science with Artificial Intelligence
### Project Title: Fraud Detection in NGO Project Funding, A Data-Driven Study on the Pakistan NGO Dataset
### Batch: VIII


### CHAPTER 1: INTRODUCTION
**1.1 Background**\
Non-Governmental Organizations (NGOs) play a vital role in delivering social services and development projects. Because of the large flows of funds and the complex chains of disbursement, NGOs are vulnerable to a variety of fraudulent activities — including exaggerated funding requests, misreporting of expenses, fictitious beneficiaries, and duplicate claims. Reliable automated tools to flag suspicious applications and transactions can assist auditors, funders, and regulators in prioritizing investigations and safeguarding public and donor funds.
This project uses a real-world dataset of NGO project applications and financial entries from Pakistan (referred to in the notebook as the Pakistan NGO Fraud Detection Dataset) to develop, evaluate, and report on data-driven fraud detection methods.

**1.2 Problem Statement**\
Manual auditing of NGO project applications and expense reports is time-consuming and often reactive. There exists a need for an automated system that can: (a) detect anomalous or likely fraudulent proposals and transactions from the historical record; (b) rank or score submissions so auditors can triage limited resources; and (c) provide interpretable indicators that support further human investigation.

**1.3 Objectives**\
The primary objectives of this project are:

1.	To perform exploratory data analysis (EDA) on the Pakistan NGO dataset and to document typical patterns and anomalies.
2.	To engineer features that capture suspicious behaviour in funding requests and project attributes (examples: funding per capita, ratios, and other derived metrics).
3.	To implement and evaluate machine learning models for fraud detection — including unsupervised anomaly detection (autoencoder) and supervised classification where labels exist.
4.	To present a working pipeline that preprocesses the data, trains models, computes detection scores, and lists top suspected fraudulent cases for human review.
5.	To assess model performance using standard metrics (precision, recall, F1, AUC, where applicable) and discuss limitations and future work.

**1.4 Scope**\
This study is limited to the dataset provided in the notebook and the features contained therein. The analysis and modelling focus on:

- Project application records and financial variables (e.g., Requested_Amount_PKR, Legitimate_Estimate_PKR, Funding_per_Capita, etc.).
- Categorical attributes such as NGO_Name, Applicant_Name, City_Town_Village, and qualitative audit indicators (Fraud_Type, Fraud_Indicator).
- Machine learning experiments performed in the notebook (unsupervised autoencoder and supervised classification experiments where labels are available).
The project does not include (in this deliverable) operational deployment, integration with live funder systems, or a formal human-in-the-loop auditing UI. Those remain as recommended next steps.

**1.5 Expected Outcomes**\
By the completion of this stage, the project aims to deliver:

- A cleaned and documented dataset ready for modelling.
- Feature engineering steps and a reproducible preprocessing pipeline.
- One or more fraud detection models (notably an autoencoder-based anomaly detector implemented in Keras) and, where labels permit, supervised classifiers.
- Quantitative evaluation (classification report/reconstruction error analyses) and a ranked list of top suspected fraudulent NGO records for manual auditing.
- A concise recommendations section describing operationalization, monitoring, and further improvements.

**1.6 Beneficiaries**\
Primary beneficiaries include:\
- Audit teams for NGOs and donor organizations (who can triage flagged cases).
- Funding agencies and grant-makers (better risk management of grant disbursements).
- Regulatory bodies concerned with transparency and accountability in NGO spending.
- Researchers and data scientists working on fraud detection in the non-profit sector.

### CHAPTER 2: LITERATURE REVIEW
**Note: The notebook contains EDA and references to the dataset; the literature review below is written to match academic style and to situate the chosen methods in prior work.**\
**2.1 General**\
Fraud detection literature spans supervised classification, unsupervised anomaly detection, and hybrid strategies. In financial and transactional domains, techniques commonly used include logistic regression, decision trees, ensemble methods (Random Forests, XGBoost), and neural networks. In contexts where labeled fraudulent examples are scarce or expensive to obtain (a common problem in NGO and nonprofit audit data), unsupervised and semi-supervised methods — such as isolation forests, one-class SVMs, and autoencoders — are frequently adopted to detect anomalous patterns. Autoencoders learn a compressed representation of normal data; records with high reconstruction error are considered anomalous and warrant inspection.
Domain literature also emphasizes careful feature engineering (ratios, per-capita figures, temporal features), handling class imbalance, and evaluation using precision/recall and ranking metrics (precision at K, average precision) rather than accuracy alone.

**2.2 Literature Collection**\
Representative findings and methodological inspirations drawn from the literature include:
- Anomaly detection with autoencoders: Using reconstruction error to detect outliers in structured financial datasets; works demonstrate good performance when genuine anomalous patterns differ substantially from normal behavior.
- Feature engineering importance: Studies show that domain features such as funding per beneficiary, discrepancy between requested and estimated cost, and unusual transaction timing are highly predictive of fraudulent claims.
- Evaluation under label scarcity: Research advocates for human-in-the-loop validation of top-ranked anomalies, and for using unsupervised techniques to reduce the search space for auditors.
- Hybrid approaches: Combining unsupervised anomaly scores with supervised classifiers (when labels exist) can yield better calibrated and interpretable results.

References to specific papers and canonical sources should be added in the final deliverable (References section) using recent articles on autoencoder-based fraud detection, imbalance learning, and NGO audit case studies.
 
### CHAPTER 3: METHODOLOGY
**3.1 Introduction**\
The methodological pipeline implemented in the notebook follows standard data science practice:
1.	Data ingestion and initial inspection (reading CSV files, printing head() and shape).
2.	Data cleaning and pre-processing (handling missing values, type conversions, date parsing).
3.	Exploratory data analysis — distributions, counts of Fraud_Type, and visualizations to understand feature distributions.
4.	Feature engineering — creation of derived variables such as Funding_per_Capita, ratios between requested and legitimate estimates, and encoding of categorical fields where required.
5.	Model development using both unsupervised and supervised approaches.
6.	Evaluation and ranking of suspected fraudulent records.

**3.1.1 Literature Review (method summary)**\
The choice of models and evaluation strategy is informed by the literature summarized above: when labels are present, supervised metrics such as precision, recall, and F1 are computed; when labels are scarce or noisy, an autoencoder is trained to reconstruct normal examples and measure reconstruction error as an anomaly score.

**3.1.2 Questionnaire Design**\
The notebook and dataset are primarily archival/transactional and do not include primary questionnaire data. Thus, no questionnaire instrument was designed as part of the data analysis. For future work involving human stakeholders (auditors, NGO staff), a short questionnaire could be designed to capture procedural information (e.g., procurement practices, expense approval chains) that may improve feature richness.

**3.1.3 Data Analysis**\
Data analysis steps performed (as implemented in the notebook):
- Data profiling: Inspect column names and types. Identify key fields such as Date, Applicant_Name, NGO_Name, Project_Name, Requested_Amount_PKR, Legitimate_Estimate_PKR, Funding_per_Capita, Fraud_Type, and Fraud_Indicator.
- Missing value handling: Missing values in numeric derived columns (e.g., Funding_per_Capita) were filled with zero or handled appropriately. (Notebook contains a FutureWarning for an operation; such warnings were noted during preprocessing.)
- Visualization: Histograms and bar plots to show the distribution of Fraud_Type, counts by NGO_Name and City_Town_Village, and other exploratory plots.
- Feature scaling: Standardization/normalization applied before neural network training (autoencoder).
- Modeling: Implementation of a Keras-based autoencoder to learn the normal behavior; computation of reconstruction error for anomaly scoring. In addition, supervised classification experiments were performed where labels were used (classification report included — see results).
- Ranking: The notebook prints a ranked list of top suspected fraudulent NGO entries (“Top Detected Fraudulent NGOs”) based on anomaly/reconstruction scores and/or classifier predictions.

**3.2 Study Aim**\
To develop a reproducible data science pipeline that can: (1) flag anomalous NGO project funding requests, (2) provide a ranked shortlist of suspicious records for auditor review, and (3) evaluate performance on labeled samples where available.

**3.3 Questionnaire Design**\
(Not applicable to the current notebook — no questionnaire used. See 3.1.2 for future suggestions.)

**3.3.1 First Questionnaire**\
N/A

**3.3.2 Second Questionnaire**\
N/A

**3.4 Data Analysis and Collection**

**3.4.1 Market Survey**\
Not applicable. The dataset is a collection of NGO transaction and application records rather than survey responses.

**3.4.2 Labour Survey**
Not applicable.

### CHAPTER 4: PROGRESS OF WORK
**4.1 Overall Project Schedule / Timeline (summary)**\
The notebook documents the data ingestion, exploratory analysis, feature engineering, modelling, and initial evaluation stages. A suggested timeline for completion (to convert the notebook into a full final report and deliverables) is:
- Week 1–2: Finalize dataset cleaning, complete EDA, and create a documented preprocessing script.
- Week 3–4: Feature engineering and baseline supervised models (if sufficient labeled data).
- Week 5: Implement and tune unsupervised models (autoencoder) and selection of anomaly threshold.
- Week 6: Produce final evaluation, extract top suspected cases, and prepare report and presentation materials.
- Week 7: Incorporate feedback, add deployment recommendations, and finalize submission.
  (Adjust timeline according to semester deadlines and supervisor feedback.)

**4.2 Progress To Date**\
**Based on the analysis of the uploaded notebook, the following progress has been achieved:**
- Dataset documented: The notebook contains a dataset description titled “Pakistan NGO Fraud Detection Dataset” and documents key columns (Applicant_Name, NGO_Name, Project_Name, Requested_Amount_PKR, Fraud_Type, Fraud_Indicator, etc.).
- EDA completed: Several plots and tables visualize distributions and counts (multiple matplotlib figures were created).
- Feature engineering: Derived features such as Funding_per_Capita were created; missing values in such fields were handled.

**Modeling:**
- A Keras autoencoder was implemented and trained (the notebook shows training output, e.g., Epoch 1/50 and final logs such as loss and val_loss — an example reported val_loss ≈ 0.0077). Reconstruction errors were computed for each sample.
- Where labels exist, a supervised classification experiment was performed; the notebook reports a classification table with a weighted average precision/recall/f1 ≈ of 0.98 on a test sample (600 records), indicating strong performance on that labeled subset. Details of train/test split and cross-validation should be included explicitly in the final write-up (and can be expanded if requested).


**- Outputs:** A list of top detected suspected fraudulent NGOs is printed in the notebook (a short-ranked extraction was produced: “🚨 Top Detected Fraudulent NGOs:” followed by records).

**4.3 Remaining Work and Challenges**

**Remaining work**\
1.	Document experimental settings fully — include explicit train/test split details, hyperparameters, random seeds, and cross-validation procedures.
2.	Threshold selection for anomaly scores — select and justify an operating point (e.g., using labeled anomalies if available or via precision@K analysis).
3.	Robust evaluation — more thorough evaluation using cross-validation and additional metrics (ROC AUC, precision at K, PR curves) and sensitivity analysis for thresholds.
4.	Explainability — add interpretable indicators for flagged records (feature contributions, example-level explanations, SHAP values) so auditors can understand why a case was flagged.
5.	Metadata & provenance — ensure dataset documentation includes source information, date ranges, and any data privacy concerns or redactions.
6.	Prepare final report document (Word/PDF) in NED interim report format, including figures and tables extracted from the notebook.
7.	Optional — test other anomaly detectors (Isolation Forest, One-Class SVM) and ensemble the results for improved robustness.

**Challenges and limitations**

- Label scarcity and quality: Fraud labels in real datasets are often noisy or incomplete; supervised performance may not reflect generalization. Heavy reliance on labels for thresholding is risky unless labels are reliable.
- Class imbalance: Fraud is typically rare; standard accuracy is not helpful — need to focus on precision/recall and ranking metrics.
- Generalizability: Models trained on historical NGO patterns may not generalize to new fraud schemes or changes in administrative procedures.
- Data quality: Missing values, inconsistent formatting, and possible duplicate records require careful cleaning to avoid model bias.
- Interpretability: Deep models (autoencoders) provide anomaly scores but limited direct interpretability; auditors require concise reasons to act on a flagged case.
________________________________________
#### Results (summary of key notebook outputs)

- Autoencoder training: Trained for multiple epochs (example training log shows Epoch 1/50 and final validation loss ~0.0104). Reconstruction errors are small for most normal records; anomalous records had higher reconstruction errors and were listed in a “Top Detected Fraudulent NGOs” output.
- Supervised classification (where labels present): A classification report printed in the notebook reported a weighted average precision, recall, and F1 ≈ of 0.98 on a labeled test set of size ~600, indicating high performance for the labeled subset. (Full confusion matrix and per-class metrics are in the notebook outputs.)
- Top detected anomalies: The notebook prints a ranked list of suspected fraudulent NGO records (table of fields including Date, Applicant_Name, NGO_Name, Requested_Amount_PKR, Reconstruction_Error, etc.) for auditor follow-up.\
(All numeric results reported above are extracted directly from the notebook outputs. Exact table values, per-class metrics, and the list of flagged records appear in the notebook and can be copied into the final document’s Tables/Figures sections.)
________________________________________
#### Conclusions and Recommendations
**Conclusions**\
The notebook demonstrates that a combination of domain feature engineering and modern machine learning (notably an autoencoder-based anomaly detector) can effectively surface suspicious NGO project funding requests. Supervised classifiers — when trained on reasonably clean labeled data — show high performance on the available test split. Unsupervised anomaly detection provides a practical approach to find previously unseen or unlabeled fraud patterns and prioritizing human review.

Recommendations
1.	Finalize evaluation and thresholding — choose operational thresholds based on audits of top-K flagged cases and desired precision (audit resource constraints).
2.	Add interpretability — compute per-record feature contributions (e.g., SHAP) to help auditors understand why a case was flagged.
3.	Human-in-the-loop validation — pilot the model with auditors and refine using their feedback; labeled results from validated cases will improve supervised learning.
4.	Deploy a triage dashboard — implement a lightweight interface (Power BI/web app) that lists flagged records, shows indicators, and supports audit workflows.
5.	Monitor model drift — regularly re-train or validate models to account for changes in submission patterns or NGO procedures.
6.	Expand dataset and features — integrate procurement records, bank confirmations, and temporal features to strengthen detection capability.
________________________________________
**Appendix (to be included in final document)**

- Dataset description (full column list and definitions) — the notebook documents columns including: Date, Applicant_Name, NGO_Name, Project_Name, Requested_Amount_PKR, Legitimate_Estimate_PKR, Funding_per_Capita, Fraud_Type, and Fraud_Indicator.
- Code references — provide key code listings: data ingestion, preprocessing steps, feature engineering, autoencoder architecture, training loop, anomaly scoring, and supervised classification snippet. (These are available in NED_Final_Year_Project_File_01.ipynb and can be appended to the final report as an Appendix or technical annex.)
- Figures and Tables — extract the notebook figures (matplotlib) and tables (pandas outputs) and insert them into the report with captions (e.g., Figure 2.1 Methodology, Figure 3.1 Distribution of Requested Amount, Table 4.1 Top 10 flagged records).
- References — add academic and practitioner articles on fraud detection, autoencoders, imbalance learning, and NGO auditing.


