import json
import sys
from pathlib import Path

def _source_to_lines(source):
    if source is None:
        return []
    if isinstance(source, str):
        return source.splitlines(keepends=True)
    return list(source)


def convertir_ipynb_a_py(archivo_ipynb):
    archivo_ipynb = Path(archivo_ipynb)

    if not archivo_ipynb.exists():
        print(f"Error: No existe el archivo {archivo_ipynb}")
        return

    with open(archivo_ipynb, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    archivo_py = archivo_ipynb.with_suffix(".py")

    with open(archivo_py, "w", encoding="utf-8") as f:
        for celda in notebook.get("cells", []):
            cell_type = celda.get("cell_type")
            source = _source_to_lines(celda.get("source", []))

            if cell_type == "markdown":
                f.write("# %% [markdown]\n")
                for linea in source:
                    contenido = linea.rstrip("\n")
                    if contenido.strip():
                        f.write("# " + contenido.rstrip() + "\n")
                    else:
                        f.write("#\n")
                f.write("\n")

            elif cell_type == "code":
                f.write("# %%\n")
                codigo = "".join(source)
                if codigo and not codigo.endswith("\n"):
                    codigo += "\n"
                f.write(codigo)
                f.write("\n")

    print(f"Convertido correctamente: {archivo_py}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python convert_from_notebooks.py archivo.ipynb")
    else:
        convertir_ipynb_a_py(sys.argv[1])