# main.py

import config
from process_manager import ProcessManager
import time
import multiprocessing

if __name__ == '__main__':
    
    multiprocessing.set_start_method('spawn', force=True)
    
    print("===========================================================")
    print("===       Road Detection and Damage Analysis System       ===")
    print("===========================================================")
    
    start_time = time.time()
    manager = ProcessManager(config)
    manager.run()
    end_time = time.time()
    
    elapsed_seconds = end_time - start_time
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    
    print("\n===========================================================")
    print(f"🎉 Analysis Complete! (Time taken: {minutes}min {seconds}sec)")
    print(f"(Total: {elapsed_seconds:.2f} seconds)")
    print("===========================================================")