import os

def generate_structure(directory):
    structure = []
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 4 * (level)
        structure.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for file in files:
            structure.append(f"{subindent}{file}")
    return "\n".join(structure)

if __name__ == "__main__":
    project_dir = './'  # Specify your project directory here
    structure_file = 'structure.txt'

    new_structure = generate_structure(project_dir)

    with open(structure_file, 'w', encoding='utf-8') as file:
        file.write(new_structure)

    print(f"{structure_file} updated successfully.")