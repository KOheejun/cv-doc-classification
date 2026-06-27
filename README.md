# 의료문서 이미지 17-class 분류

의료 청구 관련 문서 이미지를 17개 클래스로 분류하는 경진대회 프로젝트입니다.

**결과: F1 0.9357, Mid LB 3위 / 15팀**

## 핵심 접근

### 모델 앙상블
- ConvNeXt-Small + Swin-Small 앙상블 (가중치 0.68 / 0.32)
- Group K-Fold 검증으로 일반화 성능 확보
- Augraphy 기반 문서 특화 증강 (스캔 노이즈, 잉크 번짐 등)

### OCR 타이브레이커
유사 클래스인 3번(통원/외래확인서) ↔ 7번(입퇴원확인서)은 앙상블 확률로 구분이 어려워, 이미지에서 OCR로 텍스트를 추출해 키워드 기반 후처리 규칙으로 최종 클래스를 결정했습니다.

```
OCR 키워드 규칙:
- '입원', '퇴원' → 클래스 7 (입퇴원확인서)
- '통원', '외래', '진료' (조건부) → 클래스 3 (통원/외래확인서)
```

## 파일 구성

```
cv-doc-classification/
├── notebooks/
│   ├── 01_baseline.ipynb               # 초기 베이스라인
│   ├── 02_clean_pipeline.ipynb         # 검은 테두리 보정 + 증강 개선
│   └── 03_final_ensemble_multimodal.ipynb  # 최종 앙상블 + OCR 멀티모달
└── scripts/
    ├── make_ocr_37_final_v2.py         # 3↔7 OCR 후처리 스크립트
    └── ocr_tiebreak_3_7_oof.py         # OOF 기반 OCR 규칙 검증
```

## 기술 스택

| 구분 | 내용 |
|------|------|
| 모델 | ConvNeXt-Small, Swin-Small (timm) |
| 증강 | Augraphy, Albumentations |
| 검증 | Group K-Fold (문서 단위) |
| 후처리 | EasyOCR 기반 키워드 타이브레이커 |
| 실험 관리 | Weights & Biases |

## 실험 에서 배운 것

- 문서 이미지는 일반 이미지 증강(색상 변환 등)보다 **스캔 노이즈 증강(Augraphy)** 이 더 효과적
- 유사 클래스 구분에는 모델 확률보다 **도메인 지식 기반 규칙** 이 더 신뢰도 높음
- K-Fold 앙상블보다 **모델 아키텍처 다양성 앙상블** 이 성능 향상에 기여
