import os
import nbformat

# --- Config ---
FOLDER_PATH = "./"  # folder containing your notebooks
OUTPUT_FILE = "combined_notebooks.txt"
SEPARATOR = '\n' + '-'*60 + '\n'  # separator between notebooks

# --- Find notebooks ---
notebooks = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".ipynb")]
if not notebooks:
    print("No notebooks found.")
    exit()

print(f"Found {len(notebooks)} notebook(s):")
for i, nb_file in enumerate(notebooks, 1):
    print(f"{i}. {nb_file}")

# --- Open the output file ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    # --- Process notebooks ---
    for nb_file in notebooks:
        choice = input(f"\nInclude '{nb_file}' in the combined text? (y/n): ").lower()
        if choice != "y":
            print(f"Skipping '{nb_file}'...")
            continue

        print(f"Processing '{nb_file}'...")
        nb_path = os.path.join(FOLDER_PATH, nb_file)
        with open(nb_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Add notebook header
        header = f"Notebook: {nb_file}"
        outfile.write(header + "\n")
        outfile.write(SEPARATOR)

        # Convert cells
        for cell in nb.cells:
            if cell.cell_type == "markdown":
                outfile.write("# --- Markdown ---\n")
                for line in cell.source.splitlines():
                    outfile.write(f"# {line}\n")
                outfile.write("\n")
            elif cell.cell_type == "code":
                outfile.write("# --- Code ---\n")
                outfile.write(cell.source + "\n\n")
                # Add outputs if they exist
                for output in cell.get("outputs", []):
                    if output.output_type == "stream":
                        outfile.write("# --- Output ---\n")
                        for line in output.text.splitlines():
                            outfile.write(f"# {line}\n")
                    elif output.output_type == "execute_result":
                        outfile.write("# --- Result ---\n")
                        if "text/plain" in output.data:
                            for line in output.data['text/plain'].splitlines():
                                outfile.write(f"# {line}\n")
                    elif output.output_type == "error":
                        outfile.write("# --- Error ---\n")
                        for line in output.traceback:
                            outfile.write(f"# {line}\n")
                outfile.write("\n")

        # Separator after each notebook
        outfile.write(SEPARATOR)

print(f"\n✅ All selected notebooks have been combined into '{OUTPUT_FILE}'.")