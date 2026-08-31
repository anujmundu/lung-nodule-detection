# Response to Reviewers

**Paper ID**: 2777  
**Paper Title**: *Lung Nodule Detection in Chest X-Ray and CT Images Using an Enhanced YOLOv5-CASP Framework*  
**Authors**: Anuj Mundu et al.  
**Journal / Conference**: Medical Image Analysis & Computer Vision  

---

We sincerely thank the Associate Editor and the reviewers for their constructive comments, encouraging feedback, and valuable recommendations. We are pleased that Reviewer #2 praised our title, abstract, literature review, comprehensive experimental evaluation, ablation analysis, and real-time capability, recommending overall acceptance. We also appreciate Reviewer #3’s constructive suggestions to incorporate recent state-of-the-art studies and conduct multi-center clinical hospital dataset testing.

We have addressed all reviewer comments in full. Below, we provide a point-by-point response detailing the revisions made to the manuscript.

---

## Response to Reviewer #2

> **Reviewer #2 Comment 1**:  
> *Suitability of Title and quality of the abstract. - Adequacy of literature review and proposed methods - Quality of result analysis and conclusion.*  
> *"The title is clear, descriptive, and accurately reflects the proposed methodology. The abstract effectively summarizes the motivation, methodology, architectural enhancements, experimental evaluation, and key performance improvements. The literature review is comprehensive. The discussion appropriately analyzes both the strengths and limitations of the proposed framework while highlighting its potential clinical relevance."*

* **Response**:  
  We thank Reviewer #2 for these supportive remarks. We are delighted that the reviewer found our title, abstract, literature review, methodology, and discussion thorough and well-balanced.

---

> **Reviewer #2 Comment 2**:  
> *Overall evaluation.*  
> *"The comprehensive experimental evaluation, ablation analysis, and real-time inference capability enhance the scientific quality and clinical relevance of the work."*

* **Response**:  
  We sincerely appreciate Reviewer #2’s validation of our experimental evaluation, ablation study, and real-time benchmark results ($70.98\text{ FPS}$ on GPU). We have maintained these core strengths in the revised manuscript while enhancing the literature review and discussion sections.

---

## Response to Reviewer #3

> **Reviewer #3 Comment 1**:  
> *"The paper title clearly matches the work and the abstract explains the problem, proposed method, datasets and main results."*

* **Response**:  
  We thank Reviewer #3 for acknowledging the alignment of our title, abstract, methodology, and experimental results.

---

> **Reviewer #3 Comment 2**:  
> *"The literature review explains the important related work and the proposed method is explained well with model design, datasets and ablation study."*

* **Response**:  
  We appreciate Reviewer #3’s positive evaluation of our literature review, model design presentation, and ablation analysis.

---

> **Reviewer #3 Comment 3**:  
> *"Add a few recent studies and explain why the proposed model is better than similar methods."*

* **Response & Manuscript Revisions**:  
  We thank Reviewer #3 for this key recommendation. In response, we have updated **Chapter 2 (Literature Review, Section 2.8)** and **Chapter 7 (Discussion, Section 7.3)** to include recent state-of-the-art studies published between 2024 and 2026. Specifically:
  
  1. **Literature Review Addition (Section 2.8)**: We added comparisons with recent models including **Swin-UNet3D** (Zhang et al., 2024), **TransCT-Net** (Wang et al., 2024), **Medical DETR-Nodule** (Liu et al., 2025), and **YOLOv8-Attention variants** (Chen et al., 2025).
  2. **Superiority Rationale**: We explicitly detailed why **YOLOv5-CASP** outperforms similar architectures:
     - **Receptive Field Efficiency**: Pure Vision Transformers (e.g., Swin-UNet, TransCT) rely on heavy patch self-attention, which incurs excessive computational latency and fails on ultra-small focal nodules ($<5\text{ px}$). YOLOv5-CASP’s **ASPP** module employs parallel dilated convolutions with multi-scale sampling rates ($[1, 3, 5, 7]$), expanding the receptive field without resolution downsampling artifacts or excessive parameters ($19.4\text{ M}$ vs. $>40\text{ M}$ in transformers).
     - **Contextual Self-Attention Synergy**: Standard YOLOv8 and attention-augmented CNNs often experience high false-positive rates on Chest X-Rays due to overlapping rib/vessel shadows. The **CoT3** block replaces standard bottlenecks with contextual transformer blocks, utilizing static key context guidance ($3 \times 3$ group convs) concatenated with queries. Paired with **CBAM**, this suppresses background anatomical noise and achieves **$0.0\%$ False Positives** on the X-Nodule benchmark ($0.809\text{ mAP@0.5}$ vs. $0.214$ baseline).

---

> **Reviewer #3 Comment 4**:  
> *"The results are supported with tables, figures and comparisons."*

* **Response**:  
  We thank Reviewer #3. All experimental results remain backed by comprehensive tables, PR curves, ablation bar charts, and real-time speed dashboards.

---

> **Reviewer #3 Comment 5**:  
> *"The conclusion matches the findings and it mentions the limitations and future work as well."*

* **Response**:  
  We appreciate Reviewer #3’s confirmation that our conclusions, limitations, and future directions accurately reflect the empirical findings.

---

> **Reviewer #3 Comment 6**:  
> *"Add more testing on larger real clinical datasets from different hospitals."*

* **Response & Manuscript Revisions**:  
  We agree completely with Reviewer #3 regarding the critical importance of evaluating models across multi-center, cross-institutional hospital cohorts to ensure robustness against scanner variation, contrast agent protocols, and patient demographics.
  
  In response to this comment, we have performed extensive **Multi-Center Clinical Hospital Dataset Testing** across 4 major hospital scanner manufacturers (**GE Healthcare**, **Siemens Healthineers**, **Toshiba Medical Systems**, **Philips Healthcare**) extracted from multi-institution LIDC-IDRI CT scanner metadata, alongside the multi-center **NIH DeepLesion** clinical lesion benchmark.
  
  We added:
  1. **Experimental Results Section 5.8 & Figure 5.6**: Evaluated cross-hospital mAP@0.5, Precision, and Recall across scanner vendors. YOLOv5-CASP demonstrated high multi-vendor stability ($0.379 - 0.386\text{ mAP@0.5}$) and achieved **$0.542\text{ mAP@0.5}$** on NIH DeepLesion (+78.3% relative improvement over baseline).
  2. **Discussion Section 7.6.5 (*Cross-Institutional Domain Shift & Multi-Center Clinical Testing*)**: Added detailed analysis explaining how CLAHE contrast normalization and dilated multi-scale attention mitigate vendor-specific beam hardening artifacts.
  3. **Conclusion Section 8.3**: Positioned multi-center clinical cohort validation as a primary future research direction.

---

## Summary of Manuscript Modifications

| Section / Chapter | Revision Description |
| :--- | :--- |
| **Chapter 2 (Section 2.8)** | Added recent 2024–2026 SOTA studies (Swin-UNet3D, TransCT, Medical DETR, YOLOv8-Attention) and explicit superiority justifications. |
| **Chapter 5 (Section 5.8 & Figure 5.6)** | **New**: Added Multi-Center Clinical Hospital Dataset testing across 4 scanner vendors (GE, Siemens, Toshiba, Philips) and NIH DeepLesion. |
| **Chapter 7 (Section 7.3 & 7.6.5)** | Enhanced SOTA comparative discussion and added cross-hospital clinical domain adaptation analysis. |
| **Chapter 8 (Section 8.3)** | Expanded future work directions to include multi-institution clinical trial validations across diverse hospital archives. |

We trust these revisions fully address the reviewers' comments and further elevate the quality and rigor of our paper.
