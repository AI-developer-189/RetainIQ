import json

with open(r"E:\RetainIQ\FreshBasket_Churn_EDA (2).ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell.get("source", []))
        print(f"=== Cell {i} ===")
        print(source[:800])
        print()