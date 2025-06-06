import os
import shutil
def clear_folders_and_delete_file(folders_to_clear = None, file_to_delete =None):
    # 刪除每個資料夾的內容
    
    if folders_to_clear is not None:
        for folder in folders_to_clear:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)  # 刪除檔案或連結
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)  # 刪除目錄
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
    if file_to_delete is not None:
        # 刪除指定的檔案
        if os.path.exists(file_to_delete):
            try:
                os.remove(file_to_delete)
            except Exception as e:
                print(f'Failed to delete {file_to_delete}. Reason: {e}')
if __name__ == '__main__':
    
    if 1 : # 刪除資料
        # 定義要刪除內容的資料夾路徑和指定檔案
        folders_to_clear = [
            'data/measure_data',
            'data/rule_data',
            'output/backtesting',
            'output/report',
            'output/model',
            'output/score'
        ]
        # 使用 os.path.abspath 獲取當前腳本的絕對路徑
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 將每個文件夾的相對路徑轉換為絕對路徑
        folders_to_clear = [os.path.normpath(os.path.join(base_path, folder)) for folder in folders_to_clear]

        clear_folders_and_delete_file(folders_to_clear = folders_to_clear)
        
    if 1:# 刪除回測和報表
        folders_to_clear = [
            'output/backtesting',
            'output/report',
        ]
        file_to_delete = 'QuantStockPicker/report/bigtable.csv'
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folders_to_clear = [os.path.join(os.path.normpath(base_path) ,os.path.normpath(folder) ) for folder in folders_to_clear]

        file_to_delete = os.path.join(os.path.normpath(base_path) ,os.path.normpath(file_to_delete))
        clear_folders_and_delete_file(folders_to_clear = folders_to_clear, file_to_delete = file_to_delete)
