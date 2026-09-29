import torch
from transformers import AutoModel
import gc

def run_oom_benchmark(device="cuda"):
    print("--- [벤치마킹] GeneFormer 어텐션 OOM 테스트 시작 ---")
    
    # 1. 사전 훈련된 GeneFormer 로드 (Hugging Face)
    print("Loading Pre-trained Geneformer from Hugging Face...")
    try:
        # 모델을 불러오고 메모리에 적재
        geneformer = AutoModel.from_pretrained("ctheodoris/Geneformer").to(device)
        geneformer.eval() # 벤치마킹을 위해 평가 모드 설정 (Dropout 등 비활성화)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. 테스트할 시퀀스 길이 점진적 증가
    # 2048(기본)부터 시작하여 전체 유전체 스케일(30000)까지 N을 증가시킴
    test_lengths = [2048, 4096, 8192, 16384, 30000] 
    
    print("\n--- OOM 임계점 측정 (O(N^2) 병목 실증) ---")
    
    # 그레디언트 계산 비활성화 (순수 Forward Pass 연산 및 메모리 측정)
    with torch.no_grad(): 
        for N in test_lengths:
            try:
                # 3. 더미 입력 데이터 생성 (Batch=1)
                # Geneformer의 vocab 사이즈(25424) 내의 랜덤 랭크 토큰
                dummy_input_ids = torch.randint(0, 25424, (1, N)).to(device)
                dummy_attention_mask = torch.ones((1, N)).to(device)
                
                # 측정 전 VRAM 캐시 비우기
                torch.cuda.empty_cache()
                gc.collect()
                
                # 연산 시작 전 할당된 VRAM 용량
                mem_before = torch.cuda.memory_allocated(device) / (1024 ** 3)
                
                # 4. Forward Pass 실행 (Attention Matrix 생성)
                output = geneformer(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
                
                # 연산 완료 후 할당된 VRAM 용량
                mem_after = torch.cuda.memory_allocated(device) / (1024 ** 3)
                
                # 순수하게 모델 연산에 추가로 사용된 VRAM 계산
                mem_used = mem_after - mem_before
                
                print(f"[성공] Sequence Length: {N:>5} | Added VRAM Usage: {mem_used:.2f} GB")
                
                # 다음 루프를 위해 변수 삭제 및 메모리 확보
                del output, dummy_input_ids, dummy_attention_mask
                
            except torch.cuda.OutOfMemoryError:
                # VRAM이 부족하여 연산이 실패하는 정확한 지점(임계점) 포착
                print(f"\n[OOM 붕괴!] Sequence Length: {N}에서 VRAM 폭발 발생.")
                print(f"-> Transformer Attention의 O(N^2) 한계점이 증명되었습니다.")
                torch.cuda.empty_cache()
                break # 실험 종료
            except Exception as e:
                print(f"Unexpected error during sequence length {N}: {e}")
                break

if __name__ == "__main__":
    # 장치 확인 (GPU 환경 필수)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")
    
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")
        run_oom_benchmark(device=device)
    else:
        print("CUDA is not available. OOM Benchmark requires a GPU environment.")