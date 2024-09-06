import os
import re
import subprocess
import sys

project_dir = './'

def extract_imports_from_file(filepath):
    print(f"Extracting imports from file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
        imports = re.findall(r'^\s*(?:import|from)\s+([\w.]+)', content, re.MULTILINE)
        return imports

def get_unique_imports(directory):
    print(f"Scanning directory for Python files: {directory}")
    unique_imports = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Found Python file: {file_path}")
                imports = extract_imports_from_file(file_path)
                unique_imports.update(imports)
    print(f"Unique imports found: {unique_imports}")
    return unique_imports

def get_package_versions(packages):
    print(f"Fetching versions for packages: {packages}")
    package_versions = []
    for package in packages:
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'show', package], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                version_match = re.search(r'^Version:\s+(.+)', result.stdout, re.MULTILINE)
                if version_match:
                    version = version_match.group(1)
                    package_versions.append(f"{package}=={version}")
                    print(f"Found version for {package}: {version}")
                else:
                    print(f"Version not found for {package}")
            else:
                print(f"Failed to fetch version for {package}. Error: {result.stderr}")
        except Exception as e:
            print(f"Error fetching version for {package}: {e}")
    print(f"Package versions: {package_versions}")
    return package_versions

if __name__ == "__main__":
    print("Starting script...")
    unique_imports = get_unique_imports(project_dir)
    package_versions = get_package_versions(unique_imports)

    with open('requirements.txt', 'w') as req_file:
        print("Writing package versions to requirements.txt...")
        for package_version in package_versions:
            req_file.write(package_version + '\n')

    print('requirements.txt file created successfully.')