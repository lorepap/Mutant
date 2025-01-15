import psutil
import time
import argparse
import csv
import json
import signal
import sys

# Global variable to store CPU data
cpu_data = []

def monitor_cpu_usage(pid, interval=1, duration=10, output_file="cpu_usage.csv"):
    """
    Monitors CPU usage for the given process ID and its threads.
    Handles process termination gracefully.
    """
    global cpu_data
    try:
        process = psutil.Process(pid)
        print(f"Monitoring CPU usage for PID {pid}...")
        start_time = time.time()

        while time.time() - start_time < duration:
            # Check if the process is still running
            if not process.is_running():
                print(f"Process {pid} has terminated.")
                break

            # Monitor CPU usage
            main_cpu = process.cpu_percent(interval=interval)
            thread_cpus = [{"thread_id": t.id, "cpu_time": t.user_time + t.system_time} for t in process.threads()]
            timestamp = time.time()
            cpu_data.append({"timestamp": timestamp, "main_cpu": main_cpu, "threads": thread_cpus})
            print(f"Timestamp: {timestamp}, Main CPU Usage: {main_cpu}%, Threads: {thread_cpus}")

        print(f"Monitoring finished. Saving data to {output_file}...")
        save_data(cpu_data, output_file)
        print(f"CPU usage data saved to {output_file}.")
    except psutil.NoSuchProcess:
        print(f"No process found with PID {pid}.")
        save_data(cpu_data, output_file)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        save_data(cpu_data, output_file)

def save_data(cpu_data, output_file):
    """Saves CPU usage data to a file."""
    if not cpu_data:
        print("No data to save.")
        return

    if output_file.endswith(".csv"):
        save_to_csv(cpu_data, output_file)
    elif output_file.endswith(".json"):
        save_to_json(cpu_data, output_file)
    else:
        print(f"Unsupported file format: {output_file}. Supported formats are .csv and .json.")

def save_to_csv(cpu_data, output_file):
    """Save CPU usage data to a CSV file."""
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "main_cpu", "thread_id", "thread_cpu_time"])
        for entry in cpu_data:
            for thread in entry["threads"]:
                writer.writerow([entry["timestamp"], entry["main_cpu"], thread["thread_id"], thread["cpu_time"]])

def save_to_json(cpu_data, output_file):
    """Save CPU usage data to a JSON file."""
    with open(output_file, mode='w') as file:
        json.dump(cpu_data, file, indent=4)

def signal_handler(sig, frame):
    """Handles script interruption (e.g., Ctrl+C) and saves data."""
    global cpu_data
    print("\nMonitoring interrupted. Saving collected data...")
    save_data(cpu_data, "cpu_usage.csv")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C

    parser = argparse.ArgumentParser(description="Monitor CPU usage of a process.")
    parser.add_argument("--pid", type=int, required=True, help="Process ID to monitor.")
    parser.add_argument("--interval", type=int, default=1, help="Sampling interval in seconds.")
    parser.add_argument("--duration", type=int, default=10, help="Total monitoring duration in seconds.")
    parser.add_argument("--output", type=str, default="cpu_usage.csv", help="File to save the CPU usage data.")
    args = parser.parse_args()

    monitor_cpu_usage(args.pid, args.interval, args.duration, args.output)
