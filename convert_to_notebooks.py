import json
import os
import uuid

def convert_py_to_notebook(py_file_path, output_notebook_path):
    """Convierte un script Python con formato de notebook (# %%) a un notebook de Jupyter compatible con Snowflake."""
    
    with open(py_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    cells = []
    current_cell = []
    current_cell_type = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Detectar inicio de celda markdown
        if line.strip() == '# %% [markdown]':
            # Guardar celda anterior si existe
            if current_cell and current_cell_type:
                cells.append({
                    'cell_type': current_cell_type,
                    'metadata': {},
                    'source': current_cell
                })
            
            # Iniciar nueva celda markdown
            current_cell = []
            current_cell_type = 'markdown'
            i += 1
            continue
        
        # Detectar inicio de celda de código
        elif line.strip() == '# %%':
            # Guardar celda anterior si existe
            if current_cell and current_cell_type:
                cells.append({
                    'cell_type': current_cell_type,
                    'metadata': {},
                    'source': current_cell
                })
            
            # Iniciar nueva celda de código
            current_cell = []
            current_cell_type = 'code'
            i += 1
            continue
        
        # Agregar línea a la celda actual
        if current_cell_type == 'markdown':
            # Remover el '#' inicial de las líneas markdown
            if line.startswith('# '):
                current_cell.append(line[2:])
            elif line.strip() == '':
                current_cell.append('')
            else:
                current_cell.append(line)
        elif current_cell_type == 'code':
            current_cell.append(line)
        else:
            # Si no hay tipo de celda definido, asumir código
            if not current_cell_type:
                current_cell_type = 'code'
            current_cell.append(line)
        
        i += 1
    
    # Agregar última celda
    if current_cell and current_cell_type:
        cells.append({
            'cell_type': current_cell_type,
            'metadata': {},
            'source': current_cell
        })
    
    # Si no se encontraron celdas, crear una celda con todo el contenido
    if not cells:
        cells.append({
            'cell_type': 'code',
            'metadata': {},
            'source': lines
        })
    
    # Crear estructura del notebook compatible con Snowflake
    notebook = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.8.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    # Ajustar celdas para compatibilidad con Snowflake
    for cell in notebook['cells']:
        # Agregar ID único (requerido por Snowflake)
        cell['id'] = str(uuid.uuid4())
        
        # Convertir source a string (Snowflake usa strings, no arrays)
        if isinstance(cell['source'], list):
            cell['source'] = '\n'.join(cell['source'])
        
        if cell['cell_type'] == 'code':
            # Metadata específica para celdas de código en Snowflake
            cell['metadata'] = {
                'language': 'python'
            }
            cell['execution_count'] = None
            cell['outputs'] = []
        elif cell['cell_type'] == 'markdown':
            # Metadata específica para celdas markdown en Snowflake
            cell['metadata'] = {
                'codeCollapsed': True
            }
    
    # Guardar notebook
    with open(output_notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"Convertido: {py_file_path} -> {output_notebook_path}")

# Convertir todos los scripts
migration_dir = 'migration'
notebooks_dir = os.path.join(migration_dir, 'notebooks')

# Archivos a convertir
scripts = [
    '01_data_validation_and_cleaning.py',
    '02_feature_store_setup.py',
    '03_hyperparameter_search.py',
    '03b_hyperparameter_search_bayesian.py',
    '04_many_model_training.py',
    '05_create_partitioned_model.py',
    '06_partitioned_inference_batch.py'
]

for script in scripts:
    py_path = os.path.join(migration_dir, script)
    notebook_name = script.replace('.py', '.ipynb')
    notebook_path = os.path.join(notebooks_dir, notebook_name)
    
    if os.path.exists(py_path):
        convert_py_to_notebook(py_path, notebook_path)
    else:
        print(f"WARNING: No encontrado: {py_path}")

print(f"\nConversion completada! Notebooks guardados en: {notebooks_dir}")
