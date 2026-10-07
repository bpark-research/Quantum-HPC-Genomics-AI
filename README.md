# Quantum-HPC-Genomics-AI
**M-A-Q (Mamba-Attention-Quantum) 아키텍처 기반 신약 타겟 발굴 벤치마킹 및 시스템 검증**

본 프로젝트는 난치 위암 전사체 데이터(RNA-seq)를 기반으로 고전 컴퓨팅의 한계를 극복하고 양자 이득을 실증하기 위한 'M-A-Q(Mamba-Attention-Quantum) 하이브리드 아키텍처'의 PoC(Proof of Concept) 코드입니다.

---
## ToDO
### 1. 실제 벤치마킹용 유전자 발현량 데이터 연동
### 2. ~~온톨로지용 그래프 데이터베이스 구축~~
### 3. QuREKA 시험
### 4. 한강 6호기 GPU 5구좌 하이브리드 병렬처리 전략 도출
### 5. phase 2, 3 코드 작성
---
## Doing
---
## Done
### 온톨로지용 그래프 데이터베이스 (RDF Knowledge Database) 구축: Neo4j 설치
### 포트포워딩: ssh -L 7474:localhost:7474 -L 7687:localhost:7687 [계정]@[IBS-Yonsei]
### 1. Gene Ontology 다운로드 (https://geneontology.org/docs/download-ontology/) 
### 2. Import to Neo4j (https://neo4j.com/labs/rdflib-neo4j/)
### 3. Neo4j Ontology 설정:
```cypher
CREATE CONSTRAINT resource_uri_unique IF NOT EXISTS
FOR (r:Resource) REQUIRE r.uri IS UNIQUE;
```
### 4. 확인
```cypher
SHOW CONSTRAINTS;
```
### 5. Python code 작성: /neo4j/bpark.ipynb
```python
# go.owl 안에 모든 네임스페이스 출력 
# 네임스페이스를 짧게 바꾸고 서버에 저장
# Neo4j에는 251,505개의 노드와 475,126개의 관계 모두 들어 감.
```
### 6. http://localhost:7474 접속 확인 (포트포워딩 먼저 해줘야 함)
---
## 🚀 Environment Setup Guide (환경 설정 가이드)

본 프로젝트는 하드웨어 환경(CUDA 지원 여부)에 따라 투트랙(Two-track)으로 코드를 실행합니다. 

### 1. Local Development (Mac/Windows)
CUDA가 없는 로컬 환경에서는 `mamba-ssm`이 설치되지 않으므로, Mamba의 동작을 단순 모사(Mock)하는 `MambaFeatureExtractor(mock=True)` 모드로 입출력 텐서 매핑 스켈레톤 코드만 테스트합니다.

```bash
# Python 3.11 가상환경 생성 및 활성화
$ conda create -n PennyLane python=3.11 -y
$ conda activate PennyLane

# 기본 의존성 패키지 설치
$ pip install -r requirements-base.txt
```

### 2. Server Execution (QuREKA / Hangang)
실제 위암 전사체 데이터 압축 및 CUDA-Q 기반 Dynamic PQC 구동은 할당받은 GPU 노드에서 컨테이너 기반(혹은 독립 가상환경)으로 실행합니다. 트랜스포머 어텐션 병목 OOM 벤치마킹을 위한 GeneFormer 및 Mamba 레이어가 포함됩니다.

```bash
# 서버 환경에서 필수 의존성 패키지 설치 (CUDA 환경 필수)
$ pip install -r requirements-hpc.txt
```

---

## 🧬 Project Execution Workflow (실행 워크플로우)

**1. [벤치마킹] 트랜스포머 어텐션 병목 임계점(OOM) 도출**
* Hugging Face의 `ctheodoris/Geneformer`를 로드하여 시퀀스 길이를 점진적으로 늘려가며($N=2048 \rightarrow 4096 \rightarrow 8192$) B200 환경에서의 VRAM 폭발 임계점을 측정합니다.

**2. [Phase 1] Mamba 기반 선형 특성 추출 ($O(N)$ 압축)**
* OOM이 발생하는 시점부터 어텐션을 걷어내고, Mamba 모델로 30,000개 이상의 유전체 시퀀스를 선형 속도로 스캔하여 127 큐비트용 특성 텐서를 초고속 압축 추출합니다. 

**3. [Phase 2] CUDA-Q 기반 Dynamic PQC 매핑 (GPU $\rightarrow$ QPU)**
* Mamba가 추출한 텐서(Q)를 양자 회로의 회전 게이트 각도($\vec{\theta}$)로 실시간 주입하는 하이브리드 스켈레톤 코드를 QuREKA 에뮬레이터에서 구동합니다.

**4. [Phase 3] 양자 측정 및 교차 어텐션(Cross-Attention) 필터링**
* 힐베르트 공간으로 팽창된 양자 상태 결과를 다시 GPU로 반환하여 소프트 측정(Soft Measurement)으로 필터링하고 최적의 타겟을 도출합니다.
