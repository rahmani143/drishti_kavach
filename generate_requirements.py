import ast
import os
import sys

# Standard library modules to ignore (Python 3.10+)
STANDARD_LIBS = sys.stdlib_module_names if hasattr(sys, 'stdlib_module_names') else {
    'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'contextlib',
    'copy', 'csv', 'datetime', 'enum', 'fnmatch', 'functools', 'glob', 'hashlib',
    'html', 'http', 'importlib', 'inspect', 'io', 'json', 'logging', 'math',
    'multiprocessing', 'os', 'pathlib', 'pickle', 'queue', 'random', 're',
    'shutil', 'signal', 'socket', 'sqlite3', 'ssl', 'string', 'subprocess',
    'sys', 'threading', 'time', 'tkinter', 'traceback', 'types', 'typing',
    'unittest', 'urllib', 'uuid', 'warnings', 'xml', 'zipfile'
}

def get_imports_from_file(file_path):
    """Extracts top-level import names from a python file."""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=file_path)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:  # Ignore relative imports
                    imports.add(node.module.split('.')[0])
    except (SyntaxError, UnicodeDecodeError):
        # Skip files that cannot be parsed
        pass
    return imports

def generate_requirements(project_dir, output_file='requirements.txt'):
    """Scans the directory and writes unique external imports to a file."""
    detected_modules = set()
    local_modules = set()

    # Walk through directory to find all Python files and local module names
    for root, dirs, files in os.walk(project_dir):
        # Skip virtual environments and hidden directories
        dirs[:] = [d for d in dirs if d not in ('.venv', 'venv', 'env', '.git', '__pycache__', 'drishti_kavach_env')]
        
        for file in files:
            if file.endswith('.py'):
                # Track local file names so we don't mistake them for external packages
                module_name = file[:-3]
                local_modules.add(module_name)
                
                file_path = os.path.join(root, file)
                detected_modules.update(get_imports_from_file(file_path))

    # Filter out standard libraries and local modules
    external_requirements = detected_modules - STANDARD_LIBS - local_modules

    # Filter out empty strings if any
    external_requirements = {req for req in external_requirements if req}

    # Write to requirements.txt
    with open(output_file, 'w', encoding='utf-8') as f:
        for req in sorted(external_requirements):
            f.write(f"{req}\n")
            
    print(f"Created {output_file} with {len(external_requirements)} dependencies.")

if __name__ == "__main__":
    # Run in the current working directory
    current_directory = os.getcwd()
    generate_requirements(current_directory)