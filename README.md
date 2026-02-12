# UTItoSpeech: Multimodal Diffusion ATS Skeleton

이 저장소는 TaL 코퍼스 기반으로 아래 연구 가설을 빠르게 검증하기 위한 **연구용 코드 스켈레톤**입니다.

- 입력: Ultrasound tongue video (U), Lip video (L), optional text/speaker conditioning.
- 출력: mel-spectrogram (vocoder 입력).
- 핵심: Dual-stream encoder + cross-modal fusion + conditional diffusion decoder.

## 구현된 구성

1. **Dual-stream encoders**
   - U encoder: frame CNN + temporal Transformer
   - L encoder: frame CNN + temporal Transformer
2. **Fusion module**
   - `concat` (baseline)
   - `cross_attention` (U↔L)
3. **Conditional diffusion mel decoder**
   - UNet-like residual blocks with cross-attention conditioning
   - Speaker embedding FiLM conditioning
4. **통합 모델**
   - U only / L only / U+L 모두 하나의 프레임워크에서 처리
5. **TaL 데이터셋 메타데이터/스플릿 유틸**
   - TaL80 speaker ID 파서 (`01fi`, `03mn`)
   - utterance ID 파서 (`002_cal`, `029_xsil`)
   - 코퍼스 디렉토리 스캔 (`TaL1/core/day*`, `TaL80/core/<speaker>`)
   - prompt/speaker-disjoint split 유틸
   - spontaneous `.lab` 세그먼트 파서

## TaL 디렉토리 가정

```text
/TaL1
  /core/day1/*.txt|wav|sync|ult|param|mp4(.lab optional)
/TaL80
  /core/01fi/*.txt|wav|sync|ult|param|mp4
```

`discover_tal_utterances(...)`는 utterance별로 위 6개 핵심 파일(`txt`,`wav`,`sync`,`ult`,`param`,`mp4`)이 모두 있는지 검사합니다.

## 빠른 시작

```bash
python examples_shape_check.py
python examples_tal_dataset_layout.py
```

> `torch`가 필요합니다(모델 실행용).  
> `examples_tal_dataset_layout.py`는 torch 없이도 실행됩니다.

## 연구 설계 팁 반영 포인트

- **TaL80**: `speaker_disjoint_split`로 화자 독립 실험 구성
- **TaL1**: `day*` 세션 단위 분석 + 적응
- **Leakage 완화**: `prompt_disjoint_split`로 train/test tag 분리(예: train=`xaud,xsil`, test=`sil`)
- **spo 처리**: `.lab`를 이용한 segment 단위 학습

## 다음 단계 제안

- tal-tools 기반 `.ult` → 프레임 변환 파이프라인 연결
- waveform/mel 추출 및 sync alignment 로더 구현
- vocoder (HiFi-GAN 등) 연결
- speaker adaptation 루틴(embedding-only / LoRA) 추가
- diffusion scheduler 및 objective (v-pred, EDM 등) 확장
