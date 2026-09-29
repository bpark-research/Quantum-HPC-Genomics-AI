import torch
import torch.nn as nn

# mamba-ssm 패키지 가용 여부 확인 (서버 환경: True / CUDA 없는 로컬: False)
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("⚠️ [Notice] mamba-ssm이 설치되지 않은 환경입니다. 로컬 테스트용 Mock 모드로 동작합니다.")

class MambaFeatureExtractor(nn.Module):
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2, n_qubits=127):
        super().__init__()
        self.d_model = d_model
        
        # 1. 생물학적 발현량 임베딩 레이어: 순수 스칼라 발현량을 256차원 문맥 벡터로 확장
        self.expression_embedding = nn.Linear(1, d_model)
        
        # 2. Mamba 선택적 상태 공간 모델 (O(N) 선형 스캔 코어)
        if HAS_MAMBA:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            # 로컬 테스트용 선형 레이어 대체재
            self.mamba = nn.Linear(d_model, d_model)
            
        # 3. 양자 칩(QPU) 제어용 텐서 투영 레이어 (127 큐비트 웜스타트 각도 매핑용)
        self.quantum_projection = nn.Linear(d_model, n_qubits)

    def forward(self, raw_expression):
        """
        raw_expression shape: [Batch, Sequence Length, Features] 
        (예: [1, 30000, 1] -> 환자 1명, 3만 개 유전자, 각 유전자 발현량 1개)
        """
        # Step 1. 임베딩 변환: [B, L, 1] -> [B, L, d_model]
        x = self.expression_embedding(raw_expression)
        
        # Step 2. Mamba 선택적 스캔 (O(N) 복잡도로 노이즈 망각 및 핵심 시그널 압축)
        if HAS_MAMBA:
            x = self.mamba(x)
        else:
            x = self.mamba(x)  # Mock Linear 연산
            
        # Step 3. 시퀀스의 최종 상태값(마지막 유전자 토큰 처리 후)을 127차원 텐서(Q)로 추출
        # 최종 출력 shape: [Batch, n_qubits] (예: [1, 127])
        q_tensor = self.quantum_projection(x[:, -1, :])
        
        return q_tensor

if __name__ == "__main__":
    # 장치 설정 (CUDA 지원 시 GPU 사용, 아니면 CPU/MPS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"🚀 Phase 1 Execution Target Device: {device}")
    print(f"==================================================")
    
    # 난치 위암 전사체 데이터 모사 (Batch=1, Seq_Len=30000, Features=1)
    batch_size = 1
    seq_len = 30000  # TCGA 위암 환자 3만 개 유전자 전사체 스케일
    
    print(f"Loading Mock RNA-seq Data (Sequence Length: {seq_len:,} genes)...")
    mock_rna_seq = torch.rand(batch_size, seq_len, 1).to(device) * 100
    
    # 모델 초기화 및 장치 적재
    extractor = MambaFeatureExtractor(d_model=256, n_qubits=127).to(device)
    extractor.eval()
    
    # Forward Pass 실행 (O(N) 선형 스캔 압축)
    with torch.no_grad():
        q_feature_tensor = extractor(mock_rna_seq)
        
    print("-" * 50)
    print(f"✅ [Phase 1 완료]")
    print(f" - 입력 RNA-seq 데이터 형태: {mock_rna_seq.shape}")
    print(f" - 추출된 양자 특징 텐서(Q) 형태: {q_feature_tensor.shape}")
    print(f" - 결과: 3만 개의 유전자 시퀀스가 O(N) 시간에 127차원 텐서로 압축되었습니다.")
    print("==================================================")