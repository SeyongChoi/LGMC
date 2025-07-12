# LGMC

LGMC는 Python 및 Cython으로 작성된 프로젝트로, 핵생성(nucleation), 그리고 기둥 습윤(pillared wetting)과 관련된 계산을 수행합니다.

## 설치 방법
```bash
git clone https://github.com/SeyongChoi/LGMC.git
cd LGMC
pip install -e .


## 폴더 및 파일 구조

| 폴더/파일           | 설명                                   |
|---------------------|----------------------------------------|
| `configs/`          | 환경설정 파일들이 위치한 폴더           |
| `lgmc/`             | 주요 라이브러리 소스코드               |
| `nucleation/`       | 핵생성 관련 코드                        |
| `pillared_wetting/` | 기둥 습윤 관련 코드                    |
| `setup.py`          | 패키지 설치 및 빌드 스크립트            |
| `.gitignore`        | Git에서 무시할 파일 목록                |



