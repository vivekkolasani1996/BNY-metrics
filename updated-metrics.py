import datetime
import csv
import os
import concurrent.futures
from runai.configuration import Configuration
from runai.api_client import ThreadedApiClient
from runai.runai_client import RunaiClient
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import multiprocessing
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CLUSTER_UUID = "712e7936-5b8a-444a-b5f5-d59078bd4d12"

class Measurement(BaseModel):
    type: str
    labels: Optional[dict] = Field(default=None)
    values: list[dict]

class WorkloadMetric(BaseModel):
    type: str
    is_static_value: bool = Field(default=False)
    conversion_factor: float = Field(default=1.0)

    def calculate(self, measurement: Measurement) -> tuple[float, float, float]:
        total = 0.0
        peak = 0.0
        total_time = 0.0
        timestamp_prev = None

        for granular in measurement.values:
            timestamp = datetime.datetime.fromisoformat(granular["timestamp"])
            if timestamp_prev:
                time_diff = (timestamp - timestamp_prev).total_seconds()
                total_time += time_diff

                value = float(granular["value"])
                if not self.is_static_value:
                    total += value * time_diff
                else:
                    total += value * time_diff

                peak = max(peak, value)
            timestamp_prev = timestamp

        weighted_avg = (total / total_time) if total_time > 0 else 0
        total_hours = total / 3600 * self.conversion_factor
        
        return total_hours, peak, weighted_avg

def get_time_windows(start_date, end_date, desired_resolution=15):
    total_duration = (end_date - start_date).total_seconds()
    samples_per_window = 1000
    window_duration = samples_per_window * desired_resolution
    
    windows = []
    current_start = start_date
    
    while current_start < end_date:
        window_end = min(current_start + datetime.timedelta(seconds=window_duration), end_date)
        windows.append((current_start, window_end))
        current_start = window_end
    
    return windows

def fetch_project_gpu_quota(client, project_id):
    try:
        path = f"/v1/k8s/clusters/{CLUSTER_UUID}/projects/{project_id}/metrics"
        response = client.api_client.call_api(
            method="GET",
            resource_path=path,
            query_params={"start": int(datetime.datetime.now().timestamp()) - 86400, 
                         "end": int(datetime.datetime.now().timestamp())},
            response_type="object"
        )
        data = response[0]['data']
        return data.get("gpuQuota", "Unknown")
    except Exception as e:
        print(f"Error fetching GPU quota: {e}")
        return "Unknown"

def process_time_window(client, workload_id: str, window_start: datetime, window_end: datetime, metrics_types: list) -> Dict[str, Any]:
    try:
        metrics_response = client.workloads.workloads.get_workload_metrics(
            workload_id=workload_id,
            start=int(window_start.timestamp()),
            end=int(window_end.timestamp()),
            metric_type=metrics_types,
            number_of_samples=1000,
        )
        return get_response_data(metrics_response)
    except Exception as e:
        print(f"Error processing window {window_start} to {window_end}: {e}")
        return {}

def process_workload(client, workload: dict, start_date: datetime, end_date: datetime, metrics_types: list, metrics_config: dict) -> Dict[str, Any]:
    workload_id = workload.get('id')
    workload_name = workload.get('name', 'Unknown')
    print(f"\nProcessing workload: {workload_name} (ID: {workload_id})")
    
    if not workload_id:
        return {}

    time_windows = get_time_windows(start_date, end_date)
    print(f"Processing {len(time_windows)} time windows")

    metrics = {
        "gpu_hours": 0.0,
        "gpu_allocated": 0.0,
        "gpu_allocated_hours": 0.0,
        "memory_hours": 0.0,
        "cpu_hours": 0.0,
        "gpu_utilization_peak": 0.0,
        "gpu_utilization_avg": 0.0,
        "cpu_utilization_peak": 0.0,
        "cpu_utilization_avg": 0.0,
        "cpu_memory_peak": 0.0,
        "cpu_memory_avg": 0.0,
        "all_measurement_timestamps": []
    }

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_window = {
            executor.submit(
                process_time_window, 
                client, 
                workload_id, 
                window_start, 
                window_end, 
                metrics_types
            ): (window_start, window_end) 
            for window_start, window_end in time_windows
        }

        for future in concurrent.futures.as_completed(future_to_window):
            try:
                metrics_data = future.result()
                if not metrics_data.get("measurements"):
                    continue

                for measurement in metrics_data["measurements"]:
                    if not measurement.get("values"):
                        continue

                    m = Measurement(**measurement)
                    metrics["all_measurement_timestamps"].extend([
                        m.values[0]["timestamp"],
                        m.values[-1]["timestamp"]
                    ])

                    if m.type in metrics_config:
                        metric = metrics_config[m.type]
                        total_hours, peak, weighted_avg = metric.calculate(m)

                        if m.type == "CPU_USAGE_CORES":
                            metrics["cpu_hours"] += total_hours
                            metrics["cpu_utilization_peak"] = max(metrics["cpu_utilization_peak"], peak)
                            metrics["cpu_utilization_avg"] = weighted_avg
                        elif m.type == "GPU_UTILIZATION":
                            metrics["gpu_hours"] += total_hours
                            metrics["gpu_utilization_peak"] = max(metrics["gpu_utilization_peak"], peak)
                            metrics["gpu_utilization_avg"] = weighted_avg
                        elif m.type == "GPU_ALLOCATION":
                            metrics["gpu_allocated"] = max(metrics["gpu_allocated"], peak)
                        elif m.type == "CPU_MEMORY_USAGE_BYTES":
                            metrics["cpu_memory_peak"] = max(metrics["cpu_memory_peak"], peak * metric.conversion_factor)
                            metrics["cpu_memory_avg"] = weighted_avg * metric.conversion_factor
                            metrics["memory_hours"] += total_hours  # Removed redundant conversion

            except Exception as e:
                print(f"Error processing window: {e}")

    if metrics["all_measurement_timestamps"]:
        actual_start = min(metrics["all_measurement_timestamps"])
        actual_end = max(metrics["all_measurement_timestamps"])
        actual_duration = (datetime.datetime.fromisoformat(actual_end) - 
                         datetime.datetime.fromisoformat(actual_start)).total_seconds() / 3600
        metrics["gpu_allocated_hours"] = actual_duration * metrics["gpu_allocated"]
        metrics["actual_start"] = actual_start
        metrics["actual_duration"] = actual_duration
    else:
        metrics["actual_duration"] = (end_date - start_date).total_seconds() / 3600
        metrics["actual_start"] = start_date.isoformat()
        metrics["gpu_allocated_hours"] = 0

    return {
        "workload": workload,
        "metrics": metrics
    }

def get_response_data(apply_result):
    try:
        response = apply_result.get()
        data = response.data
        if not isinstance(data, dict):
            print(f"Unexpected response data type: {type(data)}")
            return {}
        return data
    except Exception as e:
        print(f"Error extracting data from response: {e}")
        return {}

def fetch_departments(client):
    try:
        response = client.organizations.departments.get_departments()
        data = get_response_data(response)
        departments = data.get('departments', [])
        print(f"Total Departments Found: {len(departments)}")
        return departments
    except Exception as e:
        print(f"Error fetching departments: {e}")
        return []

def fetch_projects_with_departments(client):
    try:
        departments_response = client.organizations.departments.get_departments()
        departments_data = get_response_data(departments_response)
        departments = departments_data.get('departments', [])
        
        department_id_to_name = {dept['id']: dept['name'] for dept in departments}
        print(f"Department ID mapping: {department_id_to_name}")
        
        try:
            response = client.projects.get_projects()
            data = get_response_data(response)
        except AttributeError:
            response = client.organizations.projects.get_projects()
            data = get_response_data(response)
        
        projects = data.get('projects', [])
        print(f"Total Projects Found: {len(projects)}")
        
        project_to_department = {}
        for project in projects:
            project_name = project.get('name', 'Unknown')
            parent_id = project.get('parentId')
            
            department = department_id_to_name.get(parent_id, 'Unknown')
            if department == 'Unknown' and project.get('parent', {}).get('name'):
                department = project['parent']['name']
                
            project_to_department[project_name] = department
        
        print("Project-Department Mapping: ", project_to_department)
        
        print("\nProjects Data:")
        for project in projects:
            project_name = project.get('name', 'Unknown')
            gpu_quota = fetch_project_gpu_quota(client, project.get('id', ''))
            department = project_to_department.get(project_name, 'Unknown')
            print(f"Project: {project_name}, Department: {department}, GPU Quota: {gpu_quota}")
        
        return project_to_department, projects
    
    except Exception as e:
        print(f"Error in fetch_projects_with_departments: {e}")
        return {}, []

def main():
    client_id = os.getenv("CLIENT_ID", "test")
    client_secret = os.getenv("CLIENT_SECRET", "Mp7aeAknS1T5cM5OjWLiy2zBqxgiMsYO")
    base_url = os.getenv("BASE_URL", "https://vivek-bny.runailabs-cs.com")
    output_dir = os.getenv("OUTPUT_DIR", "/workspace")

    config = Configuration(
        client_id=client_id,
        client_secret=client_secret,
        runai_base_url=base_url,
        verify_ssl=False
    ) 

    client = RunaiClient(ThreadedApiClient(config))

    project_to_department, projects_data = fetch_projects_with_departments(client)
    if not project_to_department:
        print("No project-department mapping found.")
        return

    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=7)
    print(f"Analyzing data from {start_date.isoformat()} to {end_date.isoformat()}")

    try:
        response = client.workloads.workloads.get_workloads()
        response_data = get_response_data(response)
        workloads_data = response_data.get('workloads', [])
        print(f"Total Workloads Found: {len(workloads_data)}")
    except Exception as e:
        print(f"Error fetching workloads: {e}")
        return

    date_format = "%m-%d-%y"
    start_str = start_date.strftime(date_format)
    end_str = end_date.strftime(date_format)

    metrics_types = [
        "GPU_ALLOCATION",
        "GPU_UTILIZATION",
        "GPU_MEMORY_USAGE_BYTES",
        "CPU_REQUEST_CORES",
        "CPU_USAGE_CORES",
        "CPU_MEMORY_USAGE_BYTES"
    ]

    metrics_config = {
        "GPU_ALLOCATION": WorkloadMetric(type="GPU_ALLOCATION", is_static_value=True),
        "CPU_REQUEST_CORES": WorkloadMetric(type="CPU_REQUEST_CORES", is_static_value=True),
        "CPU_USAGE_CORES": WorkloadMetric(type="CPU_USAGE_CORES"),
        "GPU_UTILIZATION": WorkloadMetric(type="GPU_UTILIZATION"),
        "CPU_MEMORY_USAGE_BYTES": WorkloadMetric(
            type="CPU_MEMORY_USAGE_BYTES",
            conversion_factor=1/(1024**3)  # Convert bytes to GB
        ),
        "GPU_MEMORY_USAGE_BYTES": WorkloadMetric(
            type="GPU_MEMORY_USAGE_BYTES",
            conversion_factor=1/(1024**3)
        ),
    }

    allocation_filename = os.path.join(output_dir, f"project_allocations_{start_str}_to_{end_str}.csv")
    utilization_filename = os.path.join(output_dir, f"utilization_metrics_{start_str}_to_{end_str}.csv")

    os.makedirs(output_dir, exist_ok=True)

    allocation_headers = [
        "Department", "Project", "Project Allocated GPUs", "Allocated GPU - Peak",
        "Allocated GPU - Avg", "CPU Memory (GB) - Peak", "CPU Memory (GB) - Avg",
        "CPU (# Cores) - Peak", "CPU (# Cores) - Avg"
    ]

    utilization_headers = [
        "Department", "Project", "User", "Job Name", "GPU Hours", "Memory Hours (GB)",
        "CPU Hours", "GPU Allocated", "GPU Utilization % - Peak", "GPU Utilization % - Average",
        "CPU Utilization (Cores) - Peak", "CPU Utilization (Cores) - Average",
        "CPU Memory (GB) - Peak", "CPU Memory (GB) - Average"
    ]

    project_data = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_workload = {
            executor.submit(
                process_workload, 
                client, 
                workload, 
                start_date, 
                end_date, 
                metrics_types, 
                metrics_config
            ): workload 
            for workload in workloads_data
        }

        with (
            open(allocation_filename, 'w', newline='') as alloc_file,
            open(utilization_filename, 'w', newline='') as util_file,
        ):
            alloc_writer = csv.DictWriter(alloc_file, fieldnames=allocation_headers)
            util_writer = csv.DictWriter(util_file, fieldnames=utilization_headers)
            
            alloc_writer.writeheader()
            util_writer.writeheader()

            for future in concurrent.futures.as_completed(future_to_workload):
                workload = future_to_workload[future]
                try:
                    result = future.result()
                    if not result:
                        continue

                    workload = result["workload"]
                    metrics = result["metrics"]
                    project_name = workload.get('projectName', 'Unknown')
                    department = project_to_department.get(project_name, 'Unknown') 

                    util_writer.writerow({
                        "Department": department,
                        "Project": project_name,
                        "User": workload.get('submittedBy', 'Unknown'),
                        "Job Name": workload.get('name', 'Unknown'),
                        "GPU Hours": f"{metrics['gpu_allocated_hours']:.2f}",
                        "Memory Hours (GB)": f"{metrics['memory_hours']:.2f}",
                        "CPU Hours": f"{metrics['cpu_hours']:.2f}",
                        "GPU Allocated": f"{metrics['gpu_allocated']:.2f}",
                        "GPU Utilization % - Peak": f"{metrics['gpu_utilization_peak']:.2f}",
                        "GPU Utilization % - Average": f"{metrics['gpu_utilization_avg']:.2f}",
                        "CPU Utilization (Cores) - Peak": f"{metrics['cpu_utilization_peak']:.2f}",
                        "CPU Utilization (Cores) - Average": f"{metrics['cpu_utilization_avg']:.2f}",
                        "CPU Memory (GB) - Peak": f"{metrics['cpu_memory_peak']:.2f}",
                        "CPU Memory (GB) - Average": f"{metrics['cpu_memory_avg']:.2f}"
                    })

                    if project_name not in project_data:
                        project_data[project_name] = {
                            "Department": department,
                            "Project": project_name,
                            "Project Allocated GPUs": 0,
                            "Allocated GPU - Peak": 0,
                            "Allocated GPU - Avg": 0,
                            "CPU Memory (GB) - Peak": 0,
                            "CPU Memory (GB) - Avg": 0,
                            "CPU (# Cores) - Peak": 0,
                            "CPU (# Cores) - Avg": 0,
                            "count": 0
                        }

                    pd = project_data[project_name]
                    pd["Project Allocated GPUs"] += metrics["gpu_allocated"]
                    pd["Allocated GPU - Peak"] = max(pd["Allocated GPU - Peak"], metrics["gpu_allocated"])
                    pd["Allocated GPU - Avg"] = (pd["Allocated GPU - Avg"] * pd["count"] + metrics["gpu_allocated"]) / (pd["count"] + 1)
                    pd["CPU Memory (GB) - Peak"] = max(pd["CPU Memory (GB) - Peak"], metrics["cpu_memory_peak"])
                    pd["CPU Memory (GB) - Avg"] = (pd["CPU Memory (GB) - Avg"] * pd["count"] + metrics["cpu_memory_avg"]) / (pd["count"] + 1)
                    pd["CPU (# Cores) - Peak"] = max(pd["CPU (# Cores) - Peak"], metrics["cpu_utilization_peak"])
                    pd["CPU (# Cores) - Avg"] = (pd["CPU (# Cores) - Avg"] * pd["count"] + metrics["cpu_utilization_avg"]) / (pd["count"] + 1)
                    pd["count"] += 1

                except Exception as e:
                    print(f"Error processing workload {workload.get('name', 'Unknown')}: {e}")

            for project_name, data in project_data.items():
                row_data = data.copy()
                del row_data["count"]
                for key in row_data:
                    if isinstance(row_data[key], (int, float)):
                        row_data[key] = f"{float(row_data[key]):.2f}"
                alloc_writer.writerow(row_data)

    print(f"\nProject allocations written to: {allocation_filename}")
    print(f"Utilization metrics written to: {utilization_filename}")

if __name__ == "__main__":
    main()
