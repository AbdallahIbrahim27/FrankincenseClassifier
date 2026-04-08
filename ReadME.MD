# 🌿 Frankincense Resin Classifier

موديل Image Classification يصنف حالة راتنج البخور على شجرة Boswellia.

## Classes
| Class | المعنى |
|-------|--------|
| `no_resin` | الشجرة بدون راتنج |
| `not_ready` | راتنج طازج غير ناضج |
| `ready` | راتنج جاهز للحصاد ✅ |

---

## Pipeline الكامل

```
step1_collect_data.py   ← جمع الصور من Google + Bing
step2_clean_data.py     ← تنظيف (تالف / مكرر / صغير)
step3_augment.py        ← رفع الصور لـ 1000 لكل class
step4_train.py          ← تدريب EfficientNetB0
step5_evaluate.py       ← تقييم + Confusion Matrix
step6_predict.py        ← تجربة على صور جديدة
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## التشغيل خطوة بخطوة

```bash
# 1. جمع الصور (~30-60 دقيقة)
python step1_collect_data.py

# 2. تنظيف (راجع الصور يدوياً بعدها!)
python step2_clean_data.py

# 3. Augmentation
python step3_augment.py

# 4. التدريب (~20-40 دقيقة على GPU)
python step4_train.py

# 5. التقييم
python step5_evaluate.py

# 6. التجربة على صورة جديدة
python step6_predict.py --image my_tree.jpg
python step6_predict.py --folder my_images/
```

---

## Dataset Structure

```
dataset/
├── raw/           ← الصور المجمّعة خام
│   ├── no_resin/
│   ├── not_ready/
│   └── ready/
├── clean/         ← بعد التنظيف
└── augmented/     ← بعد الـ augmentation (1000 لكل class)

models/
├── best_model_phase1.keras
├── best_model_phase2.keras
├── frankincense_classifier_final.keras
└── class_indices.json

evaluation/
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── wrong_predictions.png
└── classification_report.txt
```

---

## Model Architecture

```
EfficientNetB0 (ImageNet pretrained)
    ↓ GlobalAveragePooling2D
    ↓ BatchNormalization
    ↓ Dense(256, relu)
    ↓ Dropout(0.4)
    ↓ Dense(128, relu)
    ↓ Dropout(0.3)
    ↓ Dense(3, softmax)
```

### Training Strategy
- **Phase 1** (10 epochs): Base frozen, train head only — lr=1e-3
- **Phase 2** (20 epochs): Unfreeze last 30 layers, fine-tune — lr=1e-4

---

## نصائح مهمة

1. **التنظيف اليدوي مهم جداً** — بعد step2، افتح كل folder وشيل الصور الغلط يدوياً
2. **لو عندك GPU** — التدريب هياخد ~20 دقيقة. بدون GPU ~2 ساعة
3. **لو الـ accuracy وقفت عند 70%** — زود الصور أو حسّن التنظيف
4. **Google Colab** — ممكن تشغل step4 و step5 على Colab مجاناً

---

## Expected Results (تقديري)
- Accuracy: **85-92%** بعد fine-tuning
- أكبر confusion: بين `not_ready` و `ready` (متشابهين بصرياً)