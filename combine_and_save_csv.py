import pandas as pd
import os

def combine_and_save_csv_files(directory_path, output_file):
    """
    주어진 디렉토리 내의 모든 CSV 파일을 하나의 데이터프레임으로 결합하고 저장하는 함수.
    
    :param directory_path: CSV 파일이 있는 디렉토리 경로
    :param output_file: 결합된 CSV 파일을 저장할 경로
    :return: 결합된 데이터프레임
    """
    # 모든 CSV 파일을 담을 리스트
    csv_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.csv')]
    
    # 빈 데이터프레임 초기화
    combined_df = pd.DataFrame()
    
    # CSV 파일을 하나씩 읽어 결합
    for file in csv_files:
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    # 결합된 데이터프레임을 CSV 파일로 저장
    combined_df.to_csv(output_file, index=False)
    
    return combined_df

# 사용 예시
directory_path = 'phishing'  # CSV 파일들이 있는 디렉토리 경로

output_file = 'combined_dataset.csv'  # 결합된 CSV 파일을 저장할 경로
combined_df = combine_and_save_csv_files(directory_path, output_file)
pd.set_option('display.max_columns', None)
print(combined_df)
# 결합된 데이터프레임 출력
print(combined_df.head())
