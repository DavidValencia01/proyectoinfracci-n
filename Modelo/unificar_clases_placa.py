from pathlib import Path
import shutil

# Rutas de los directorios de etiquetas
label_dirs = [
    Path('PeruPlateNumbers.v3i.yolov8/train/labels'),
    Path('PeruPlateNumbers.v3i.yolov8/valid/labels'),
    Path('PeruPlateNumbers.v3i.yolov8/test/labels'),
]

def unify_labels():
    for label_dir in label_dirs:
        if not label_dir.exists():
            print(f"No existe el directorio: {label_dir}")
            continue
        for label_file in label_dir.glob('*.txt'):
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                if line.strip() == '':
                    continue
                parts = line.strip().split()
                if parts[0] == '1':
                    parts[0] = '0'  # Cambia clase 1 a 0
                new_lines.append(' '.join(parts) + '\n')
            with open(label_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"Procesado: {label_file}")

# Actualizar data.yaml

def update_data_yaml():
    yaml_path = Path('PeruPlateNumbers.v3i.yolov8/data.yaml')
    backup_path = yaml_path.with_suffix('.yaml.bak')
    shutil.copy(yaml_path, backup_path)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith('nc:'):
            new_lines.append('nc: 1\n')
        elif line.strip().startswith('names:'):
            new_lines.append("names: ['Placa']\n")
        else:
            new_lines.append(line)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Actualizado {yaml_path} (backup en {backup_path})")

if __name__ == '__main__':
    unify_labels()
    update_data_yaml()
    print('¡Unificación de clases completada!') 