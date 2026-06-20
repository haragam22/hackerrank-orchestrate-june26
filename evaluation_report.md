# Evaluation Report

## Summary
- **Total claims evaluated:** 20
- **Gate Check:** ✅ GATE PASSED (≥70%)

## Metrics

| Metric | Score |
|---|---|
| Claim Status Accuracy | **80.0%** |
| Severity Accuracy | 55.0% |
| Issue Type Accuracy | 60.0% |
| Object Part Accuracy | 90.0% |
| Evidence Standard Met Acc. | 90.0% |
| Valid Image Accuracy | 85.0% |
| Risk Flags (F1 Score) | 61.3% |

## Error Analysis (Claim Status)

Detailed breakdown of claims where the predicted status did not match the ground truth:

### Claim: `user_002`
- **Status Match:** ❌ Predicted `contradicted`, Expected `supported`
- **Issue Type:** Predicted `none`, Expected `scratch`
- **Risk Flags:** Predicted `damage_not_visible`, Expected `none`

**User Claim Transcript:**
> Customer: Parking lot mein meri car ko scrape lag gaya. | Support: Aap kis type ka damage report karna chahte hain? | Customer: Front side par mark aa gaya hai, bumper ke upar. | Support: Light damage hai ya body par scratch? | Customer: Light theek hai, front bumper par scratch hai. Photos upload kar diye hain.

**Predicted Justification (Agent 4):**
> img_2 shows no visible signs of a scratch on the front bumper. The surface appears smooth and intact, contradicting the claim. Image quality is clear and does not affect the assessment.

**Expected Justification (Ground Truth):**
> The close-up image shows a visible scratch on the claimed front bumper.

🔍 **Improvement Insight:** The Vision Model completely missed the damage that was expected to be seen. This points to a hallucination in Agent 3 (Damage Assessor). Consider improving the few-shot examples or prompt instructions for the claimed object part.

---

### Claim: `user_012`
- **Status Match:** ❌ Predicted `contradicted`, Expected `supported`
- **Issue Type:** Predicted `none`, Expected `dent`
- **Risk Flags:** Predicted `low_light_or_glare;damage_not_visible;user_history_risk;manual_review_required`, Expected `none`

**User Claim Transcript:**
> Customer: My laptop bag fell from a chair and hit the floor. | Support: What damage did you notice? | Customer: One corner of the laptop has a dent now. | Support: Does the laptop still work? | Customer: It works, but the outer corner is damaged. I added two photos.

**Predicted Justification (Agent 4):**
> img_2 clearly shows the corner of the laptop, and no dent is visible. The surface appears undamaged, contradicting the claim. The second image, despite having some glare, met the evidence standard for assessment.

**Expected Justification (Ground Truth):**
> The image set supports the claim because the corner dent is visible in the close-up.

🔍 **Improvement Insight:** The Vision Model completely missed the damage that was expected to be seen. This points to a hallucination in Agent 3 (Damage Assessor). Consider improving the few-shot examples or prompt instructions for the claimed object part.

---

### Claim: `user_032`
- **Status Match:** ❌ Predicted `contradicted`, Expected `not_enough_information`
- **Issue Type:** Predicted `none`, Expected `unknown`
- **Risk Flags:** Predicted `cropped_or_obstructed;damage_not_visible;user_history_risk;manual_review_required`, Expected `cropped_or_obstructed;damage_not_visible;manual_review_required`

**User Claim Transcript:**
> Customer: The item I ordered was not inside the box. | Support: Did the package look opened when you received it? | Customer: I checked it after delivery and could not find the product inside. | Support: What are you asking us to verify? | Customer: Please verify that the contents are missing from the package.

**Predicted Justification (Agent 4):**
> img_1 shows numerous crumpled items within the package, indicating that the contents are not missing. However, the lack of an exterior image to confirm the package's integrity and the presence of contents prevents a definitive resolution. The claim is contradicted based on the available evidence.

**Expected Justification (Ground Truth):**
> The package contents are unclear, so the missing-product claim cannot be verified from the submitted images.

🔍 **Improvement Insight:** Review the prompt alignment. There may be a discrepancy between what Agent 3 considers valid damage vs the ground truth standard.

---

### Claim: `user_034`
- **Status Match:** ❌ Predicted `supported`, Expected `contradicted`
- **Issue Type:** Predicted `torn_packaging`, Expected `none`
- **Risk Flags:** Predicted `non_original_image;text_instruction_present;user_history_risk;manual_review_required`, Expected `damage_not_visible;text_instruction_present;user_history_risk;manual_review_required`

**User Claim Transcript:**
> Customer: My delivery box arrived opened. | Support: Was the package crushed or was the seal affected? | Customer: The seal area looked torn when I received it. | Support: Are you asking us to review the package condition or the item inside? | Customer: The package condition. I want the torn-open package reviewed.

**Predicted Justification (Agent 4):**
> img_1 clearly shows the seal area of the package with visible tearing and separation, indicating torn packaging. The damage is noticeable and affects the integrity of the seal. The presence of text instructions and a stock photo in the images does not affect the visual confirmation of the claim.

**Expected Justification (Ground Truth):**
> The visible package seal does not show torn-open packaging. Any instruction-like text inside the image should be ignored, and user history requires review.

🔍 **Improvement Insight:** The Vision Model hallucinated damage that does not exist. Agent 3 is being too lenient or mistaking normal wear-and-tear / reflections for damage. Tune Agent 3 to be more skeptical.

---

