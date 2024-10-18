import psutil
import subprocess
import sys
import time
import os
from datetime import datetime

# Function to monitor CPU and memory usage of a process
def monitor_utilization(pid, duration, interval=1):
    process = psutil.Process(pid)
    cpu_usages = []
    memory_usages = []
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        
        try:
            # Record CPU and memory usage at the current time
            cpu_usage = process.cpu_percent(interval=interval)
            memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            if process.status() == psutil.STATUS_ZOMBIE:
                break
            cpu_usages.append(cpu_usage)
            memory_usages.append(memory_usage)
            
            print(f"{datetime.now().strftime('[%H:%M:%S]')}::CPU-{cpu_usage:06.2f}% | MEM-{memory_usage:06.2f}MB")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            # If the process has terminated or can't be accessed, break the loop
            print(f"Process terminated or inaccessible. {e}")
            break
    
    return cpu_usages, memory_usages

# Function to run and profile a Python file
def profile_script(script_path):
    if not os.path.exists(script_path):
        print(f"Error: The file '{script_path}' does not exist.")
        return

    # Start the target Python script using subprocess
    print(f"Starting {script_path}...")
    process = subprocess.Popen([sys.executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"PID: {process.pid}")
    
    try:
        # Monitor CPU and memory usage for the duration of the process's execution
        cpu_usages, memory_usages = monitor_utilization(process.pid, duration=60, interval=1)
        
        # Wait for the process to complete
        process.wait()

        output, error = process.communicate()
        #print(output.decode())
        
        # Calculate min, max, and average CPU and memory usage
        if cpu_usages and memory_usages:
            min_cpu = min(cpu_usages)
            max_cpu = max(cpu_usages)
            avg_cpu = sum(cpu_usages) / len(cpu_usages)
            
            min_memory = min(memory_usages)
            max_memory = max(memory_usages)
            avg_memory = sum(memory_usages) / len(memory_usages)
            
            print("\nCPU and Memory Utilization Results:")
            print(f"Min CPU: {min_cpu:.2f}%, Max CPU: {max_cpu:.2f}%, Avg CPU: {avg_cpu:.2f}%")
            print(f"Min Memory: {min_memory:.2f} MB, Max Memory: {max_memory:.2f} MB, Avg Memory: {avg_memory:.2f} MB")
        else:
            print("No data recorded. Process may have exited too quickly.")

    except Exception as e:
        print(f"Error while profiling: {e}")
    finally:
        process.terminate()
        exit()

if __name__ == '__main__':
    # Ensure the script is passed a target file
    if len(sys.argv) < 2:
        print("Usage: python profiler.py <path_to_python_script>")
        sys.exit(1)

    script_path = sys.argv[1]
    
    # Call the profiling function
    profile_script(script_path)
