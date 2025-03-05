# metrics

Run:AI Resource Usage Report Generator
This script generates reports about resource utilization across projects and departments in a Run:AI environment. It extracts metrics about GPU allocation, CPU usage, and memory utilization, organizing them by project and department.


Generates two detailed CSV reports:

Project Allocation Report: Aggregated resource usage by project
Utilization Metrics Report: Detailed resource usage by individual workloads


Customizable date ranges for reporting periods
Custom file naming for output reports
SSL certificate verification bypass for development environments

Required Packages
Install the following packages using pip:

pip install pydantic
pip install urllib3
pip install aiohttp-retry
pip install aiohttp
pip install -i https://test.pypi.org/simple/ runapy==3.0.0


Once all dependencies are installed, you can run the script:

python metrics.py


Configuration
Before running the script, you need to open the metrics.py file in a text editor and modify these variables in the main() function:

# Replace these values with your Run:AI API credentials
client_id = "your_client_id"
client_secret = "your_client_secret"
base_url = "https://your-runai-instance.example.com"

# Set your preferred output directory
output_dir = os.getenv('OUTPUT_DIR', '/your/preferred/output/path')

Parameters

client_id: Your Run:AI API client ID
client_secret: Your Run:AI API client secret
base_url: The base URL of your Run:AI instance
output_dir: Directory where CSV reports will be saved (defaults to /workspace if not specified)

Running the Script

Open the metrics.py file in a text editor and modify the credential variables as shown above
Run the script:

python metrics.py

Follow the interactive prompts:

Enter the year (e.g., 2025): 2025  
Enter the month (1-12): 2
Enter the start day of the month: 1
Enter the end day of the month: 28


Choose custom filenames for your reports (or press Enter to use the default date-based names):

Default output filenames will be based on the date range.
Default allocation file: project_allocations_02-01-25_to_02-28-25.csv
Default utilization file: utilization_metrics_02-01-25_to_02-28-25.csv

Enter custom name for allocation file (or press Enter for default): q1_project_allocations
Enter custom name for utilization file (or press Enter for default): q1_utilization_metrics


Output Reports
Project Allocation Report
Utilization Report
















