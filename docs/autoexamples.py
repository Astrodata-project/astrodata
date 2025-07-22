import os

EXAMPLES_DIR = "../examples"
OUTPUT_DIR = "python_examples"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

for root, _, files in os.walk(EXAMPLES_DIR):
    for file in files:
        if file.endswith(".py") and file != "__init__.py":
            rel_dir = os.path.relpath(root, EXAMPLES_DIR)
            output_dir = os.path.join(OUTPUT_DIR, rel_dir)
            ensure_dir(output_dir)
            base_name = os.path.splitext(file)[0]
            rst_filename = base_name + ".rst"
            rst_path = os.path.join(output_dir, rst_filename)
            example_path = os.path.join(root, file)
            # Title: filename (or you could use something else)
            title = f"{rel_dir}/{file}" if rel_dir != '.' else file
            with open(rst_path, "w") as f:
                f.write(f"{title}\n{'=' * len(title)}\n\n")
                f.write(f".. literalinclude:: ../../{example_path}\n")
                f.write("   :language: python\n")
