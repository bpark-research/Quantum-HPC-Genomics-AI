import pandas as pd
import numpy as np

def calculate_hybrid_usage(nodes_per_job, jobs_per_day, duration_days):
    """
    멀티GPU 환경의 하이브리드 병렬화를 적용한 사용량 산출 분석
    """
    # 1. 한강 6호기 자원 할당 기준 (일반 공모 5구좌 신청)[cite: 1, 8]
    total_hours_per_account = 300
    num_accounts = 5
    total_budget_hours = total_hours_per_account * num_accounts # 1500 노드시간 확보[cite: 1, 8]
    
    # 2. 작업 스케줄링 변수 및 병목 제약 반영
    # 한강 6호기 Slurm 정책: 동시 작업 제출 최대 5개 제한
    max_concurrent_jobs = 5
    actual_jobs_per_day = min(jobs_per_day, max_concurrent_jobs)
    
    # 작업당 평균 실행 시간 (Hybrid 병렬화 최적화 기준)
    # Mamba 전처리(O(N))는 1시간 내외, Transformer OOM 벤치마킹은 2시간 내외로 가정하여 평균 1.5시간 산정[cite: 8, 9]
    hours_per_job = 1.5 
    
    # 3. 누적 자원 사용량 계산 (Node-Hours)
    daily_node_hours = nodes_per_job * actual_jobs_per_day * hours_per_job
    total_used_hours = daily_node_hours * duration_days
    
    # 4. 분석 결과 출력
    print("=" * 60)
    print("🚀 M-A-Q 하이브리드 파이프라인(멀티GPU) 자원 사용량 분석")
    print("=" * 60)
    print(f"[자원 확보] 신청 구좌: {num_accounts} 구좌 (총 {total_budget_hours} 노드시간)[cite: 1, 8]")
    print(f"[작업 설정] 하이브리드 병렬화 단위: {nodes_per_job} Nodes/Job")
    print(f"[작업 설정] 일일 구동 작업 수 (5개 제한 반영): {actual_jobs_per_day} Jobs/day[cite: 8]")
    print(f"[소요 시간] 작업당 평균 런타임: {hours_per_job} Hours")
    print(f"[수행 기간] 베타 테스트 집중 구동일: {duration_days} Days")
    print("-" * 60)
    print(f"📊 일일 자원 소모량: {daily_node_hours:.1f} Node-Hours/day")
    print(f"📊 총 예상 사용량: {total_used_hours:.1f} Node-Hours")
    print(f"📊 잔여 노드시간(Buffer): {total_budget_hours - total_used_hours:.1f} Node-Hours")
    print("=" * 60)
    
    if total_used_hours > total_budget_hours:
        print("⚠️ [경고] 예상 사용량이 신청한 1,500 노드시간을 초과합니다. 노드 수나 작업 빈도를 줄이세요.")
    else:
        print("✅ [적정] 5구좌(1,500 노드시간) 예산 내에서 완벽히 통제된 하이브리드 벤치마킹이 가능합니다.[cite: 1, 8]")
        
    return total_used_hours

if __name__ == "__main__":
    # 시나리오: 단일 작업 내 멀티 GPU 병렬화 효율을 극대화하기 위해 
    # 작업당 8개의 팻 노드(총 64 B200 GPU)를 하이브리드 병렬화로 묶어 할당함.
    # 동시 실행 제한을 고려해 하루 3개의 모듈화된 작업을 돌리며, 
    # 베타 테스트 기간 중 실질적인 연산 집중일을 20일로 잡고 산출함.
    calculate_hybrid_usage(nodes_per_job=8, jobs_per_day=3, duration_days=20)